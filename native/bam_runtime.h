#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bamkv {

// Status codes returned to Python; mirror IoStatus in runtime/descriptors.py.
enum IoStatusCode : int { PENDING = 0, DONE = 1, FAILED = 2 };

struct BamDeviceParams {
    std::vector<std::string> nvme_paths;
    int nvm_namespace;
    int cuda_device;
    int queue_depth;
    int num_queues;
    int64_t page_size;
    int64_t page_cache_pages;
    int64_t lba_block_size;
};

// Owns BaM Controllers + page cache + a cudaEvent handle table.
// All CUDA/BaM types live in the .cu via the opaque Impl.
class BamDevice {
public:
    explicit BamDevice(const BamDeviceParams& params);
    ~BamDevice();

    BamDevice(const BamDevice&) = delete;
    BamDevice& operator=(const BamDevice&) = delete;

    // dptr: GPU virtual address of the KV tensor; nbytes: extent size;
    // dev_offset: byte offset into the namespace (LBA-aligned);
    // stream: cudaStream_t as uintptr_t (0 => default stream).
    uint64_t submit_store(uintptr_t dptr, int64_t nbytes, int64_t dev_offset, uintptr_t stream);
    uint64_t submit_load(uintptr_t dptr, int64_t nbytes, int64_t dev_offset, uintptr_t stream);
    int poll(uint64_t handle);   // cudaEventQuery -> IoStatusCode
    int wait(uint64_t handle);   // cudaEventSynchronize -> IoStatusCode

private:
    struct Impl;
    Impl* impl_;
    // Page-region ring reservation; member so it can name the private Impl.
    static int64_t reserve_pages(Impl* d, int64_t n_pages);
};

}  // namespace bamkv
