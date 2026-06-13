#include "bam_runtime.h"

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <cmath>

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

// ---- device-side I/O, modeled on benchmarks/readwrite/main.cu:43-70 ----
// Issues one NVMe R/W per page-cache entry covering blocks_per_page LBAs.
__device__ static void rw_one_page(page_cache_d_t* pc, QueuePair* qp,
                                   uint64_t starting_lba, uint64_t n_blocks,
                                   unsigned long long pc_entry, bool is_write) {
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, is_write ? NVM_IO_WRITE : NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps) prp2 = pc->prp2[pc_entry];
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    sq_dequeue(&qp->sq, sq_pos);
    cq_dequeue(&qp->cq, cq_pos);
    put_cid(&qp->sq, cid);
}

// One thread (block) per page of the extent. page_base = first page-cache index
// holding this extent's bytes; lba_base = first LBA of the extent.
__global__ static void rw_extent_kernel(Controller** ctrls, page_cache_d_t* pc,
                                        uint64_t lba_base, uint64_t page_base,
                                        uint64_t n_pages, uint64_t blocks_per_page,
                                        uint32_t num_ctrls, bool is_write) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pages) return;
    uint32_t ctrl = (i / 32) % num_ctrls;
    uint32_t queue = (i / 32) % (ctrls[ctrl]->n_qps);
    uint64_t starting_lba = lba_base + i * blocks_per_page;
    rw_one_page(pc, (ctrls[ctrl]->d_qps) + queue, starting_lba, blocks_per_page,
                page_base + i, is_write);
}

struct BamDevice::Impl {
    BamDeviceParams params;
    std::vector<Controller*> ctrls;
    Controller** d_ctrls = nullptr;   // device array of controller pointers
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
    // Device array of controller pointers (benchmark passes Controller** to kernels).
    ck(cudaMalloc(&impl_->d_ctrls, impl_->ctrls.size() * sizeof(Controller*)), "cudaMalloc d_ctrls");
    ck(cudaMemcpy(impl_->d_ctrls, impl_->ctrls.data(), impl_->ctrls.size() * sizeof(Controller*),
                  cudaMemcpyHostToDevice), "cudaMemcpy d_ctrls");

    // Page cache — mirrors benchmarks/readwrite/main.cu:269.
    impl_->h_pc = new page_cache_t(params.page_size, params.page_cache_pages, params.cuda_device,
                                   impl_->ctrls[0][0], (uint64_t)64, impl_->ctrls);
    impl_->d_pc = (page_cache_d_t*)(impl_->h_pc->d_pc_ptr);
    impl_->base_addr = impl_->h_pc->pdt.base_addr;
    impl_->blocks_per_page = params.page_size / params.lba_block_size;
}

BamDevice::~BamDevice() {
    if (!impl_) return;
    for (auto& kv : impl_->events) cudaEventDestroy(kv.second);
    if (impl_->h_pc) delete impl_->h_pc;
    if (impl_->d_ctrls) cudaFree(impl_->d_ctrls);
    for (auto* c : impl_->ctrls) delete c;
    delete impl_;
}

// Reserve a contiguous page-region of n_pages in the ring; returns first page index.
static int64_t reserve_pages(BamDevice::Impl* d, int64_t n_pages) {
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
        int64_t n_pages = (nbytes + d->params.page_size - 1) / d->params.page_size;
        int64_t page_base = reserve_pages(d, n_pages);
        uint64_t lba_base = (uint64_t)(dev_offset / d->params.lba_block_size);
        char* pc_region = (char*)d->base_addr + page_base * d->params.page_size;
        // KV tensor (GPU) -> page-cache buffer (GPU), then flush pages to SSD.
        ck(cudaMemcpyAsync(pc_region, (void*)dptr, nbytes, cudaMemcpyDeviceToDevice, s), "memcpy store D2D");
        int threads = 256, blocks = (int)((n_pages + threads - 1) / threads);
        rw_extent_kernel<<<blocks, threads, 0, s>>>(d->d_ctrls, d->d_pc, lba_base, page_base,
                                                    n_pages, d->blocks_per_page,
                                                    (uint32_t)d->ctrls.size(), /*is_write=*/true);
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
        int64_t n_pages = (nbytes + d->params.page_size - 1) / d->params.page_size;
        int64_t page_base = reserve_pages(d, n_pages);
        uint64_t lba_base = (uint64_t)(dev_offset / d->params.lba_block_size);
        char* pc_region = (char*)d->base_addr + page_base * d->params.page_size;
        int threads = 256, blocks = (int)((n_pages + threads - 1) / threads);
        // SSD -> page-cache buffer (GPU), then page-cache -> KV tensor (GPU).
        rw_extent_kernel<<<blocks, threads, 0, s>>>(d->d_ctrls, d->d_pc, lba_base, page_base,
                                                    n_pages, d->blocks_per_page,
                                                    (uint32_t)d->ctrls.size(), /*is_write=*/false);
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

int BamDevice::poll(uint64_t handle) {
    auto* d = impl_;
    std::lock_guard<std::mutex> lk(d->mu);
    auto it = d->events.find(handle);
    if (it == d->events.end() || it->second == nullptr) return FAILED;
    cudaError_t q = cudaEventQuery(it->second);
    if (q == cudaSuccess) return DONE;
    if (q == cudaErrorNotReady) return PENDING;
    return FAILED;
}

int BamDevice::wait(uint64_t handle) {
    auto* d = impl_;
    cudaEvent_t ev;
    {
        std::lock_guard<std::mutex> lk(d->mu);
        auto it = d->events.find(handle);
        if (it == d->events.end() || it->second == nullptr) return FAILED;
        ev = it->second;
    }
    cudaError_t e = cudaEventSynchronize(ev);
    return e == cudaSuccess ? DONE : FAILED;
}

}  // namespace bamkv
