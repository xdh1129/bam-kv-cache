#include "bam_runtime.h"

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

// BaM headers (from $BAM_HOME/include) — same set the readwrite benchmark uses.
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_cmd.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <ctrl.h>

namespace bamkv {

// One thread per page of the extent. page_base = first page-cache index holding
// this extent's bytes; lba_base = first LBA of the extent. Uses the page cache's
// own device-side controller array (pc->d_ctrls, built correctly by page_cache_t)
// and the library's read_data/write_data primitives (page_cache.h:553-554) rather
// than re-implementing NVMe queue handling.
__global__ static void rw_extent_kernel(page_cache_d_t* pc,
                                        uint64_t lba_base, uint64_t page_base,
                                        uint64_t n_pages, uint64_t blocks_per_page,
                                        bool is_write) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pages) return;
    uint32_t ctrl = (uint32_t)((i / 32) % pc->n_ctrls);
    Controller* c = pc->d_ctrls[ctrl];
    uint32_t queue = (uint32_t)((i / 32) % c->n_qps);
    uint64_t starting_lba = lba_base + i * blocks_per_page;
    if (is_write)
        write_data(pc, (c->d_qps) + queue, starting_lba, blocks_per_page, page_base + i);
    else
        read_data(pc, (c->d_qps) + queue, starting_lba, blocks_per_page, page_base + i);
}

struct BamDevice::Impl {
    BamDeviceParams params;
    std::vector<Controller*> ctrls;
    page_cache_t* h_pc = nullptr;     // host-side page cache object (owns GPU buffer)
    page_cache_d_t* d_pc = nullptr;   // device page cache pointer (h_pc->d_pc_ptr)
    void* base_addr = nullptr;        // GPU page-cache data buffer (h_pc->pdt.base_addr)
    int64_t blocks_per_page = 0;
    // page-region ring allocator over [0, page_cache_pages)
    std::mutex mu;
    int64_t ring_next = 0;
    uint64_t next_handle = 1;
    std::unordered_map<uint64_t, cudaEvent_t> events;

    int64_t total_pages() const { return params.page_cache_pages; }
};

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

BamDevice::BamDevice(const BamDeviceParams& params) {
    impl_ = new Impl();
    impl_->params = params;
    ck(cudaSetDevice(params.cuda_device), "cudaSetDevice");

    // Controllers — mirrors benchmarks/readwrite/main.cu:239-241.
    for (size_t i = 0; i < params.nvme_paths.size(); ++i) {
        impl_->ctrls.push_back(new Controller(params.nvme_paths[i].c_str(), params.nvm_namespace,
                                              params.cuda_device, params.queue_depth, params.num_queues));
    }
    // Page cache — mirrors benchmarks/readwrite/main.cu:269. Its constructor builds
    // the device-side controller array (pdt.d_ctrls), reachable as pc->d_ctrls in kernels.
    impl_->h_pc = new page_cache_t(params.page_size, params.page_cache_pages, params.cuda_device,
                                   impl_->ctrls[0][0], (uint64_t)64, impl_->ctrls);
    impl_->d_pc = (page_cache_d_t*)(impl_->h_pc->d_pc_ptr);
    impl_->base_addr = impl_->h_pc->pdt.base_addr;
    impl_->blocks_per_page = params.page_size / params.lba_block_size;
}

BamDevice::~BamDevice() {
    if (!impl_) return;
    for (auto& kv : impl_->events) {
        if (kv.second) cudaEventDestroy(kv.second);
    }
    if (impl_->h_pc) delete impl_->h_pc;
    for (auto* c : impl_->ctrls) delete c;
    delete impl_;
}

// Reserve a contiguous page-region of n_pages in the ring; returns first page index.
// INVARIANT: the ring does not track in-flight occupancy, so the page cache must be
// sized so that all *concurrently in-flight* extents (submitted but not yet waited)
// fit without wrapping onto each other. The v1 connector submits then waits per layer
// (one in-flight extent at a time), which satisfies this. Pipelining more extents than
// fit in page_cache_pages would let a wrapped reservation clobber a buffer still being
// DMA'd — size page_cache_pages accordingly, or add occupancy tracking before doing so.
int64_t BamDevice::reserve_pages(Impl* d, int64_t n_pages) {
    if (n_pages > d->total_pages())
        throw std::runtime_error("extent larger than page cache");
    if (d->ring_next + n_pages > d->total_pages()) d->ring_next = 0;  // wrap
    int64_t base = d->ring_next;
    d->ring_next += n_pages;
    return base;
}

uint64_t BamDevice::submit_store(uintptr_t dptr, int64_t nbytes, int64_t dev_offset, uintptr_t stream) {
    auto* d = impl_;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    std::lock_guard<std::mutex> lk(d->mu);
    try {
        ck(cudaSetDevice(d->params.cuda_device), "cudaSetDevice");
        int64_t n_pages = (nbytes + d->params.page_size - 1) / d->params.page_size;
        int64_t page_base = reserve_pages(d, n_pages);
        uint64_t lba_base = (uint64_t)(dev_offset / d->params.lba_block_size);
        char* pc_region = (char*)d->base_addr + page_base * d->params.page_size;
        // KV tensor (GPU) -> page-cache buffer (GPU), then flush pages to SSD.
        ck(cudaMemcpyAsync(pc_region, (void*)dptr, nbytes, cudaMemcpyDeviceToDevice, s), "memcpy store D2D");
        int threads = 256, blocks = (int)((n_pages + threads - 1) / threads);
        rw_extent_kernel<<<blocks, threads, 0, s>>>(d->d_pc, lba_base, page_base,
                                                    n_pages, d->blocks_per_page, /*is_write=*/true);
        ck(cudaPeekAtLastError(), "rw_extent_kernel store launch");
        cudaEvent_t ev; ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "eventCreate");
        ck(cudaEventRecord(ev, s), "eventRecord");
        uint64_t h = d->next_handle++;
        d->events[h] = ev;
        return h;
    } catch (const std::exception&) {
        uint64_t h = d->next_handle++;
        d->events[h] = nullptr;  // null event => FAILED in poll/wait
        return h;
    }
}

uint64_t BamDevice::submit_load(uintptr_t dptr, int64_t nbytes, int64_t dev_offset, uintptr_t stream) {
    auto* d = impl_;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    std::lock_guard<std::mutex> lk(d->mu);
    try {
        ck(cudaSetDevice(d->params.cuda_device), "cudaSetDevice");
        int64_t n_pages = (nbytes + d->params.page_size - 1) / d->params.page_size;
        int64_t page_base = reserve_pages(d, n_pages);
        uint64_t lba_base = (uint64_t)(dev_offset / d->params.lba_block_size);
        char* pc_region = (char*)d->base_addr + page_base * d->params.page_size;
        int threads = 256, blocks = (int)((n_pages + threads - 1) / threads);
        // SSD -> page-cache buffer (GPU), then page-cache -> KV tensor (GPU).
        rw_extent_kernel<<<blocks, threads, 0, s>>>(d->d_pc, lba_base, page_base,
                                                    n_pages, d->blocks_per_page, /*is_write=*/false);
        ck(cudaPeekAtLastError(), "rw_extent_kernel load launch");
        ck(cudaMemcpyAsync((void*)dptr, pc_region, nbytes, cudaMemcpyDeviceToDevice, s), "memcpy load D2D");
        cudaEvent_t ev; ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "eventCreate");
        ck(cudaEventRecord(ev, s), "eventRecord");
        uint64_t h = d->next_handle++;
        d->events[h] = ev;
        return h;
    } catch (const std::exception&) {
        uint64_t h = d->next_handle++;
        d->events[h] = nullptr;
        return h;
    }
}

// poll is non-destructive: it never erases the handle, so it stays queryable until
// the caller's terminal wait() reclaims the event. (The v1 connector only calls wait.)
int BamDevice::poll(uint64_t handle) {
    auto* d = impl_;
    std::lock_guard<std::mutex> lk(d->mu);
    cudaSetDevice(d->params.cuda_device);
    auto it = d->events.find(handle);
    if (it == d->events.end() || it->second == nullptr) return FAILED;
    cudaError_t q = cudaEventQuery(it->second);
    if (q == cudaSuccess) return DONE;
    if (q == cudaErrorNotReady) return PENDING;
    return FAILED;
}

// wait is terminal and called exactly once per handle: it destroys and erases the
// event afterward so the table does not grow without bound over a long run.
int BamDevice::wait(uint64_t handle) {
    auto* d = impl_;
    cudaEvent_t ev;
    {
        std::lock_guard<std::mutex> lk(d->mu);
        cudaSetDevice(d->params.cuda_device);
        auto it = d->events.find(handle);
        if (it == d->events.end() || it->second == nullptr) return FAILED;
        ev = it->second;
    }
    cudaError_t e = cudaEventSynchronize(ev);
    {
        std::lock_guard<std::mutex> lk(d->mu);
        auto it = d->events.find(handle);
        if (it != d->events.end()) {
            cudaEventDestroy(it->second);
            d->events.erase(it);
        }
    }
    return e == cudaSuccess ? DONE : FAILED;
}

}  // namespace bamkv
