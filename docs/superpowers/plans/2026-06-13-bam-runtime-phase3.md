# BaM Runtime (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real BaM-backed `BamRuntime` (native CUDA `BamDevice` + Python `NativeBamRuntime`) plus an LBA allocator, so KV payload moves GPU memory ↔ NVMe SSD with no CPU staging — without touching `BamConnectorCore` or the metadata-plane logic.

**Architecture:** A torch CUDA extension (`bam_kv_cache._native.BamDevice`) owns BaM `Controller`s + a GPU `page_cache` and performs store/load by LBA using a page-cache→`cudaMemcpyAsync` D2D path, exposing integer handles backed by `cudaEvent`. Python `NativeBamRuntime(BamRuntime)` marshals `KVTensorDescriptor`/`SsdLocation`/CUDA-stream into the native calls; `RawNamespaceAllocator` hands the metadata plane LBA-aligned byte extents over the raw namespace.

**Tech Stack:** Python 3.12, torch (CUDA on the box), pybind11 via `torch.utils.cpp_extension`, CUDA 12.3+, BaM (`libnvm`). Target box: RTX 6000 Ada (sm_89), `/dev/nvme0n1`, LBA 512.

**Spec:** `docs/superpowers/specs/2026-06-13-bam-runtime-phase3-design.md`

**Build/verify split (important):**
- Tasks 1–3 and 8 are **pure Python** — they have real local pytest steps that run on any machine (no CUDA). Keep the existing 34-test suite green.
- Tasks 4–7 (native `.h/.cu/.cpp`, `setup.py`) and Task 9 (round-trip) **compile and run only on the GPU box**. Their "verify" steps are GPU-box build + the Task 9 round-trip; they are skipped locally via `pytest.importorskip`. Do not attempt to build them on the Mac.
- Run pytest from the `bam-kv-cache/` repo root. Commit only the files each task names (the repo `.gitignore` already excludes `__pycache__`, `build/`, `*.so`, `*.egg-info`).

---

## Task 1: `BamDeviceConfig`

**Files:**
- Create: `src/bam_kv_cache/config_bam.py`
- Test: `tests/test_config_bam.py`

- [ ] **Step 1: Create `src/bam_kv_cache/config_bam.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BamDeviceConfig:
    """Hardware/runtime configuration for the native BaM device.

    Discovered per box (see the spec): on the target box nvme_paths is the BaM
    kernel-module char device, lba_block_size is the NVMe LBA format in use, and
    namespace_capacity_bytes is the raw namespace size.
    """

    nvme_paths: tuple[str, ...]       # BaM controller paths, e.g. ("/dev/libnvm0",)
    nvm_namespace: int                # NVMe namespace id (usually 1)
    cuda_device: int                  # CUDA device ordinal
    queue_depth: int                  # per-queue depth
    num_queues: int                   # NVMe queue pairs per controller
    page_size: int                    # BaM page-cache page size in bytes
    page_cache_pages: int             # number of pages in the GPU page cache
    lba_block_size: int               # NVMe LBA format size (e.g. 512 or 4096)
    namespace_capacity_bytes: int     # usable raw capacity for the allocator
```

- [ ] **Step 2: Create `tests/test_config_bam.py`**

```python
from bam_kv_cache.config_bam import BamDeviceConfig


def test_construct_and_frozen():
    cfg = BamDeviceConfig(
        nvme_paths=("/dev/libnvm0",), nvm_namespace=1, cuda_device=0,
        queue_depth=1024, num_queues=128, page_size=4096,
        page_cache_pages=65536, lba_block_size=512,
        namespace_capacity_bytes=2048408248320,
    )
    assert cfg.lba_block_size == 512
    assert cfg.nvme_paths == ("/dev/libnvm0",)
```

- [ ] **Step 3: Run the test**

Run: `python3 -m pytest tests/test_config_bam.py -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add src/bam_kv_cache/config_bam.py tests/test_config_bam.py
git commit -m "feat(config): BamDeviceConfig for native BaM device"
```

---

## Task 2: `SsdAllocator` protocol + widen metadata store hint

**Files:**
- Create: `src/bam_kv_cache/ssd/allocator.py`
- Modify: `src/bam_kv_cache/metadata/store.py` (imports + `__init__` type hint only)
- Test: `tests/test_ssd_allocator.py`

**Context:** `InProcessMetadataStore` only ever calls `object_store.allocate(nbytes)`. To let either `FileObjectStore` (Phase 1–2) or `RawNamespaceAllocator` (Task 3) back it, introduce a structural `SsdAllocator` protocol and widen the hint. No behavior change — the existing 34 tests must still pass.

- [ ] **Step 1: Create `src/bam_kv_cache/ssd/allocator.py`**

```python
from typing import Protocol, runtime_checkable

from .object_store import SsdLocation


@runtime_checkable
class SsdAllocator(Protocol):
    """Hands out byte extents on the SSD address space.

    FileObjectStore (mock/CPU) and RawNamespaceAllocator (BaM, raw namespace)
    both satisfy this. The metadata plane depends only on `allocate`.
    """

    def allocate(self, nbytes: int) -> SsdLocation: ...
```

- [ ] **Step 2: Update the import in `src/bam_kv_cache/metadata/store.py`**

Find:

```python
from ..ssd.object_store import FileObjectStore, SsdLocation
```

Replace with:

```python
from ..ssd.allocator import SsdAllocator
from ..ssd.object_store import SsdLocation
```

- [ ] **Step 3: Update the constructor hint in `src/bam_kv_cache/metadata/store.py`**

Find:

```python
    def __init__(self, object_store: FileObjectStore):
        self._store = object_store
```

Replace with:

```python
    def __init__(self, object_store: SsdAllocator):
        self._store = object_store
```

- [ ] **Step 4: Create `tests/test_ssd_allocator.py`**

```python
from bam_kv_cache.metadata.keys import build_object_key
from bam_kv_cache.metadata.store import InProcessMetadataStore
from bam_kv_cache.ssd.allocator import SsdAllocator
from bam_kv_cache.ssd.object_store import FileObjectStore, SsdLocation


class _FakeAllocator:
    """Minimal allocator satisfying SsdAllocator without a file."""

    def __init__(self):
        self._next = 0
        self._id = 0

    def allocate(self, nbytes: int) -> SsdLocation:
        loc = SsdLocation(object_id=self._id, offset=self._next, nbytes=nbytes)
        self._id += 1
        self._next += nbytes
        return loc


def test_fileobjectstore_satisfies_protocol(tmp_path):
    assert isinstance(FileObjectStore(str(tmp_path)), SsdAllocator)


def test_fake_allocator_satisfies_protocol():
    assert isinstance(_FakeAllocator(), SsdAllocator)


def test_metadata_store_works_with_fake_allocator():
    meta = InProcessMetadataStore(_FakeAllocator())
    k = build_object_key(
        model_id="m", tokenizer_id="t", token_chunk_hash="c0", layer_id=0,
        tp_rank=0, kv_dtype="float32", kv_layout="NHD", chunk_size=16,
    )
    meta.reserve_store({k: 64})
    meta.commit_store([k])
    assert k in meta.lookup_meta([k])
```

- [ ] **Step 5: Run the new test AND the full suite (no regressions)**

Run: `python3 -m pytest tests/test_ssd_allocator.py -v`
Expected: 3 passed.
Run: `python3 -m pytest -q`
Expected: all pass (38 total).

- [ ] **Step 6: Commit**

```bash
git add src/bam_kv_cache/ssd/allocator.py src/bam_kv_cache/metadata/store.py tests/test_ssd_allocator.py
git commit -m "feat(ssd): SsdAllocator protocol; metadata store depends on it"
```

---

## Task 3: `RawNamespaceAllocator`

**Files:**
- Create: `src/bam_kv_cache/runtime/lba_alloc.py`
- Test: `tests/test_lba_alloc.py`

- [ ] **Step 1: Create `src/bam_kv_cache/runtime/lba_alloc.py`**

```python
import threading

from ..ssd.object_store import SsdLocation


class RawNamespaceAllocator:
    """Bump-allocates LBA-aligned byte extents over a raw NVMe namespace.

    Satisfies the SsdAllocator protocol. `offset` is always a multiple of
    `lba_block_size`, so the native runtime can compute `lba = offset //
    lba_block_size` exactly. No file/backing store — the native BaM device does
    the actual I/O at these offsets.
    """

    def __init__(self, capacity_bytes: int, lba_block_size: int, alignment: int | None = None):
        if lba_block_size <= 0:
            raise ValueError("lba_block_size must be positive")
        self._capacity = capacity_bytes
        self._lba_block_size = lba_block_size
        # Align at least to an LBA; default to the LBA size.
        self._alignment = alignment if alignment is not None else lba_block_size
        if self._alignment % lba_block_size != 0:
            raise ValueError("alignment must be a multiple of lba_block_size")
        self._lock = threading.Lock()
        self._next_offset = 0
        self._next_object_id = 0

    def _align(self, n: int) -> int:
        a = self._alignment
        return ((n + a - 1) // a) * a

    def allocate(self, nbytes: int) -> SsdLocation:
        size = self._align(nbytes)
        with self._lock:
            offset = self._next_offset
            if offset + size > self._capacity:
                raise RuntimeError(
                    f"namespace out of space: need {size} at {offset}, capacity {self._capacity}"
                )
            object_id = self._next_object_id
            self._next_object_id += 1
            self._next_offset += size
        return SsdLocation(object_id=object_id, offset=offset, nbytes=nbytes)
```

- [ ] **Step 2: Create `tests/test_lba_alloc.py`**

```python
import pytest

from bam_kv_cache.metadata.keys import build_object_key
from bam_kv_cache.metadata.store import InProcessMetadataStore
from bam_kv_cache.runtime.lba_alloc import RawNamespaceAllocator
from bam_kv_cache.ssd.allocator import SsdAllocator


def test_satisfies_protocol():
    assert isinstance(RawNamespaceAllocator(1 << 20, 512), SsdAllocator)


def test_offsets_are_lba_aligned():
    alloc = RawNamespaceAllocator(capacity_bytes=1 << 20, lba_block_size=512)
    a = alloc.allocate(100)   # rounds up to 512
    b = alloc.allocate(100)
    assert a.offset == 0
    assert b.offset == 512
    assert a.offset % 512 == 0 and b.offset % 512 == 0
    assert a.object_id != b.object_id


def test_capacity_exhaustion_raises():
    alloc = RawNamespaceAllocator(capacity_bytes=512, lba_block_size=512)
    alloc.allocate(512)
    with pytest.raises(RuntimeError):
        alloc.allocate(1)


def test_bad_alignment_raises():
    with pytest.raises(ValueError):
        RawNamespaceAllocator(1 << 20, lba_block_size=512, alignment=300)


def test_drives_metadata_store_end_to_end():
    # The metadata plane works on top of the LBA allocator with no runtime/CUDA.
    meta = InProcessMetadataStore(RawNamespaceAllocator(1 << 20, 512))
    k = build_object_key(
        model_id="m", tokenizer_id="t", token_chunk_hash="c0", layer_id=0,
        tp_rank=0, kv_dtype="float32", kv_layout="NHD", chunk_size=16,
    )
    locs = meta.reserve_store({k: 4096})
    assert locs[k].offset % 512 == 0
    meta.commit_store([k])
    assert k in meta.lookup_meta([k])
```

- [ ] **Step 3: Run the tests**

Run: `python3 -m pytest tests/test_lba_alloc.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add src/bam_kv_cache/runtime/lba_alloc.py tests/test_lba_alloc.py
git commit -m "feat(runtime): RawNamespaceAllocator (LBA-aligned extents)"
```

---

## Task 4: Native header `bam_runtime.h`

**Files:**
- Create: `native/bam_runtime.h`

**Context (GPU box only — not compiled locally):** The header must be CUDA-free so `bindings.cpp` can include it without CUDA. `BamDevice` is declared with an opaque `Impl` pointer; the `.cu` defines the CUDA/BaM internals.

- [ ] **Step 1: Create `native/bam_runtime.h`**

```cpp
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
};

}  // namespace bamkv
```

- [ ] **Step 2: Commit (no local build — header only)**

```bash
git add native/bam_runtime.h
git commit -m "feat(native): BamDevice header (CUDA-free, opaque impl)"
```

---

## Task 5: Native implementation `bam_runtime.cu`

**Files:**
- Create: `native/bam_runtime.cu`

**Context (GPU box only):** This mirrors `bam/benchmarks/readwrite/main.cu` — the authoritative working reference for your BaM build. Construction of `Controller`s (main.cu:239-241), the `page_cache_t` (main.cu:269), the `page_cache_d_t` device pointer (main.cu:275), and the per-page `read_data`/`write_data` device functions (main.cu:43-70 pattern) come from there. **Validate every BaM API call against that file on the box**, since `page_cache`/`QueuePair` internals are BaM-version-specific. The store/load wrappers below are the contract; the kernel bodies follow the benchmark.

- [ ] **Step 1: Create `native/bam_runtime.cu`**

```cpp
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
```

- [ ] **Step 2: Commit (compiles only on the box; built in Task 7)**

```bash
git add native/bam_runtime.cu
git commit -m "feat(native): BamDevice CUDA impl (page-cache D2D path, event handles)"
```

**Validation note (do on the box during Task 9):** confirm `page_cache_t` constructor args, `pdt.base_addr`, `d_pc_ptr`, `prp1/prp2`, `get_cid/sq_enqueue/cq_poll` signatures match your `$BAM_HOME` headers and the readwrite benchmark; adjust the kernel/ctor to match if the BaM version differs.

---

## Task 6: pybind bindings `bindings.cpp`

**Files:**
- Create: `native/bindings.cpp`

- [ ] **Step 1: Create `native/bindings.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bam_runtime.h"

namespace py = pybind11;
using bamkv::BamDevice;
using bamkv::BamDeviceParams;

PYBIND11_MODULE(_native, m) {
    m.doc() = "BaM-backed KV cache data plane (native).";

    py::class_<BamDevice>(m, "BamDevice")
        .def(py::init([](std::vector<std::string> nvme_paths, int nvm_namespace, int cuda_device,
                         int queue_depth, int num_queues, int64_t page_size,
                         int64_t page_cache_pages, int64_t lba_block_size) {
                 BamDeviceParams p;
                 p.nvme_paths = std::move(nvme_paths);
                 p.nvm_namespace = nvm_namespace;
                 p.cuda_device = cuda_device;
                 p.queue_depth = queue_depth;
                 p.num_queues = num_queues;
                 p.page_size = page_size;
                 p.page_cache_pages = page_cache_pages;
                 p.lba_block_size = lba_block_size;
                 return new BamDevice(p);
             }),
             py::arg("nvme_paths"), py::arg("nvm_namespace"), py::arg("cuda_device"),
             py::arg("queue_depth"), py::arg("num_queues"), py::arg("page_size"),
             py::arg("page_cache_pages"), py::arg("lba_block_size"))
        .def("submit_store", &BamDevice::submit_store,
             py::arg("dptr"), py::arg("nbytes"), py::arg("dev_offset"), py::arg("stream"))
        .def("submit_load", &BamDevice::submit_load,
             py::arg("dptr"), py::arg("nbytes"), py::arg("dev_offset"), py::arg("stream"))
        .def("poll", &BamDevice::poll, py::arg("handle"))
        .def("wait", &BamDevice::wait, py::arg("handle"));
}
```

- [ ] **Step 2: Commit**

```bash
git add native/bindings.cpp
git commit -m "feat(native): pybind11 bindings for BamDevice"
```

---

## Task 7: Conditional native build in `setup.py`

**Files:**
- Create: `setup.py`
- Modify: `pyproject.toml` (add `[project.optional-dependencies] native`)

**Context:** With `pyproject.toml` already defining the package, `setup.py` adds the CUDA extension **only** when `BAM_KV_BUILD_NATIVE=1`, so local `pip install -e .` stays pure-Python. On the box, `BAM_KV_BUILD_NATIVE=1 BAM_HOME=/path/to/bam pip install -e .` builds `bam_kv_cache._native`.

- [ ] **Step 1: Create `setup.py`**

```python
import os

from setuptools import setup

ext_modules = []
cmdclass = {}

if os.environ.get("BAM_KV_BUILD_NATIVE") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    bam_home = os.environ.get("BAM_HOME")
    if not bam_home:
        raise RuntimeError("BAM_KV_BUILD_NATIVE=1 requires BAM_HOME to point at the built BaM repo")
    arch = os.environ.get("BAM_KV_CUDA_ARCH", "89")  # RTX 6000 Ada = sm_89

    ext_modules = [
        CUDAExtension(
            name="bam_kv_cache._native",
            sources=["native/bindings.cpp", "native/bam_runtime.cu"],
            include_dirs=[os.path.join(bam_home, "include")],
            library_dirs=[os.path.join(bam_home, "build", "lib")],
            libraries=["nvm"],
            runtime_library_dirs=[os.path.join(bam_home, "build", "lib")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
```

- [ ] **Step 2: Add the `native` extra to `pyproject.toml`**

Find:

```toml
[project.optional-dependencies]
test = ["pytest>=7"]
```

Replace with:

```toml
[project.optional-dependencies]
test = ["pytest>=7"]
native = ["pybind11>=2.10"]
```

- [ ] **Step 3: Verify local install stays pure-Python (no native build)**

Run: `python3 -m pytest -q`
Expected: all pass (43 total). `setup.py` must not attempt a CUDA build when `BAM_KV_BUILD_NATIVE` is unset.

- [ ] **Step 4: Commit**

```bash
git add setup.py pyproject.toml
git commit -m "build: conditional CUDA extension (BAM_KV_BUILD_NATIVE)"
```

---

## Task 8: `NativeBamRuntime`

**Files:**
- Create: `src/bam_kv_cache/runtime/native_bam.py`
- Test: `tests/test_native_bam_glue.py`

**Context:** `NativeBamRuntime` is pure Python and depends only on the `BamRuntime` ABC + a `device` object exposing `submit_store/submit_load/poll/wait`. The marshalling/status logic is unit-tested locally against a **fake device** (no CUDA); the real `BamDevice` is supplied on the box via `from_config`.

- [ ] **Step 1: Create `src/bam_kv_cache/runtime/native_bam.py`**

```python
from typing import Any

from .base import BamRuntime
from .descriptors import IoHandle, IoStatus, KVTensorDescriptor, SlotMapping

_STATUS = {0: IoStatus.PENDING, 1: IoStatus.DONE, 2: IoStatus.FAILED}


def _stream_ptr(stream: Any) -> int:
    """Return a cudaStream_t as int. None -> current torch CUDA stream."""
    if stream is None:
        import torch
        return torch.cuda.current_stream().cuda_stream
    if hasattr(stream, "cuda_stream"):
        return stream.cuda_stream
    return int(stream)


class NativeBamRuntime(BamRuntime):
    """BamRuntime backed by the native BaM device (bam_kv_cache._native.BamDevice).

    KV tensors must be CUDA tensors; their data_ptr() is passed to the device.
    """

    def __init__(self, device: Any):
        self._dev = device

    @classmethod
    def from_config(cls, cfg) -> "NativeBamRuntime":
        from bam_kv_cache import _native  # built only on the GPU box
        dev = _native.BamDevice(
            nvme_paths=list(cfg.nvme_paths),
            nvm_namespace=cfg.nvm_namespace,
            cuda_device=cfg.cuda_device,
            queue_depth=cfg.queue_depth,
            num_queues=cfg.num_queues,
            page_size=cfg.page_size,
            page_cache_pages=cfg.page_cache_pages,
            lba_block_size=cfg.lba_block_size,
        )
        return cls(dev)

    def submit_store(self, kv: KVTensorDescriptor, slot_mapping: SlotMapping,
                     location, stream=None) -> IoHandle:
        hid = self._dev.submit_store(kv.tensor.data_ptr(), kv.nbytes, location.offset,
                                     _stream_ptr(stream))
        return IoHandle(handle_id=hid)

    def submit_load(self, kv: KVTensorDescriptor, slot_mapping: SlotMapping,
                    location, stream=None) -> IoHandle:
        hid = self._dev.submit_load(kv.tensor.data_ptr(), kv.nbytes, location.offset,
                                    _stream_ptr(stream))
        return IoHandle(handle_id=hid)

    def poll(self, handle: IoHandle) -> IoStatus:
        st = _STATUS[self._dev.poll(handle.handle_id)]
        handle.status = st
        return st

    def wait(self, handle: IoHandle) -> IoStatus:
        st = _STATUS[self._dev.wait(handle.handle_id)]
        handle.status = st
        if st is IoStatus.FAILED:
            handle.error = "bam device reported failure"
        return st
```

- [ ] **Step 2: Create `tests/test_native_bam_glue.py`**

```python
import types

from bam_kv_cache.runtime.descriptors import IoStatus, KVTensorDescriptor, SlotMapping
from bam_kv_cache.runtime.native_bam import NativeBamRuntime
from bam_kv_cache.ssd.object_store import SsdLocation


class _FakeTensor:
    def __init__(self, ptr, nbytes):
        self._ptr = ptr
        self._n = nbytes

    def data_ptr(self):
        return self._ptr


class _FakeDevice:
    """Records calls and returns scripted statuses."""

    def __init__(self):
        self.calls = []
        self._next = 1
        self.status_for = {}

    def submit_store(self, dptr, nbytes, dev_offset, stream):
        self.calls.append(("store", dptr, nbytes, dev_offset, stream))
        h = self._next; self._next += 1
        return h

    def submit_load(self, dptr, nbytes, dev_offset, stream):
        self.calls.append(("load", dptr, nbytes, dev_offset, stream))
        h = self._next; self._next += 1
        return h

    def poll(self, h):
        return self.status_for.get(h, 0)

    def wait(self, h):
        return self.status_for.get(h, 1)


def _kv(ptr=0xCAFE, nbytes=4096):
    return KVTensorDescriptor(tensor=_FakeTensor(ptr, nbytes), layer_id=0, nbytes=nbytes)


def test_submit_store_marshals_fields():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=0, offset=8192, nbytes=4096)
    h = rt.submit_store(_kv(0xABCD, 4096), SlotMapping(0, 4096), loc, stream=7)
    assert dev.calls[0] == ("store", 0xABCD, 4096, 8192, 7)
    assert h.handle_id == 1


def test_submit_load_marshals_fields():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=1, offset=0, nbytes=4096)
    rt.submit_load(_kv(0x1000, 4096), SlotMapping(0, 4096), loc, stream=3)
    assert dev.calls[0] == ("load", 0x1000, 4096, 0, 3)


def test_poll_and_wait_status_mapping():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=0, offset=0, nbytes=4096)
    h = rt.submit_store(_kv(), SlotMapping(0, 4096), loc, stream=0)
    dev.status_for[h.handle_id] = 0
    assert rt.poll(h) is IoStatus.PENDING
    dev.status_for[h.handle_id] = 1
    assert rt.wait(h) is IoStatus.DONE


def test_wait_failure_sets_error():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=0, offset=0, nbytes=4096)
    h = rt.submit_load(_kv(), SlotMapping(0, 4096), loc, stream=0)
    dev.status_for[h.handle_id] = 2
    assert rt.wait(h) is IoStatus.FAILED
    assert h.error is not None
```

- [ ] **Step 3: Run the glue tests AND full suite**

Run: `python3 -m pytest tests/test_native_bam_glue.py -v`
Expected: 4 passed.
Run: `python3 -m pytest -q`
Expected: all pass (47 total).

- [ ] **Step 4: Commit**

```bash
git add src/bam_kv_cache/runtime/native_bam.py tests/test_native_bam_glue.py
git commit -m "feat(runtime): NativeBamRuntime glue (fake-device tested)"
```

---

## Task 9: GPU-box round-trip test + run docs

**Files:**
- Create: `tests/test_native_roundtrip.py`
- Create: `docs/phase3-gpu-box.md`

**Context:** This test runs **only on the GPU box** (skipped elsewhere). It builds the full BaM stack and round-trips a real CUDA KV tensor through the SSD.

- [ ] **Step 1: Create `tests/test_native_roundtrip.py`**

```python
import os

import pytest

torch = pytest.importorskip("torch")
_native = pytest.importorskip("bam_kv_cache._native")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
if os.environ.get("BAM_KV_RUN_HW") != "1":
    pytest.skip("set BAM_KV_RUN_HW=1 to run BaM hardware round-trip", allow_module_level=True)

from bam_kv_cache.config_bam import BamDeviceConfig
from bam_kv_cache.connector.bam_connector import BamConnectorCore, KVChunk, KVRequest
from bam_kv_cache.config import KVConfig
from bam_kv_cache.metadata.store import InProcessMetadataStore
from bam_kv_cache.runtime.descriptors import IoStatus, KVTensorDescriptor, SlotMapping
from bam_kv_cache.runtime.lba_alloc import RawNamespaceAllocator
from bam_kv_cache.runtime.native_bam import NativeBamRuntime


def _dev_cfg():
    return BamDeviceConfig(
        nvme_paths=tuple(os.environ.get("BAM_KV_NVME", "/dev/libnvm0").split(",")),
        nvm_namespace=1, cuda_device=0, queue_depth=1024, num_queues=128,
        page_size=4096, page_cache_pages=65536, lba_block_size=512,
        namespace_capacity_bytes=2048408248320,
    )


def test_runtime_roundtrip_single_extent():
    dcfg = _dev_cfg()
    rt = NativeBamRuntime.from_config(dcfg)
    alloc = RawNamespaceAllocator(dcfg.namespace_capacity_bytes, dcfg.lba_block_size)

    src = torch.arange(1024, dtype=torch.float32, device="cuda")
    nbytes = src.element_size() * src.nelement()
    loc = alloc.allocate(nbytes)

    hs = rt.submit_store(KVTensorDescriptor(src, 0, nbytes), SlotMapping(0, nbytes), loc)
    assert rt.wait(hs) is IoStatus.DONE

    dst = torch.zeros_like(src)
    hl = rt.submit_load(KVTensorDescriptor(dst, 0, nbytes), SlotMapping(0, nbytes), loc)
    assert rt.wait(hl) is IoStatus.DONE
    assert torch.equal(dst, src)


def test_connector_roundtrip_on_bam():
    dcfg = _dev_cfg()
    rt = NativeBamRuntime.from_config(dcfg)
    meta = InProcessMetadataStore(RawNamespaceAllocator(dcfg.namespace_capacity_bytes, dcfg.lba_block_size))
    kcfg = KVConfig(model_id="m", tokenizer_id="t", kv_dtype="float32", kv_layout="NHD",
                    chunk_size=16, tp_rank=0, ssd_root="")
    conn = BamConnectorCore(kcfg, meta, rt)
    layers = ["layer.0", "layer.1"]
    conn.register_kv_caches({ln: None for ln in layers})

    def mk(fill):
        return KVRequest(request_id="r", chunks=[
            KVChunk(chunk_hash="c0", layer_tensors={
                ln: (torch.arange(1024, dtype=torch.float32, device="cuda") + i + fill)
                for i, ln in enumerate(layers)})])

    src = mk(0.0)
    for ln in layers:
        conn.save_kv_layer(ln, src)
    conn.wait_for_save()

    dst = KVRequest(request_id="r2", chunks=[
        KVChunk(chunk_hash="c0", layer_tensors={ln: torch.zeros(1024, device="cuda") for ln in layers})])
    assert conn.get_num_new_matched_tokens(dst) == 16
    conn.start_load_kv(dst)
    for ln in layers:
        conn.wait_for_layer_load(ln)
    for ln in layers:
        assert torch.equal(dst.chunks[0].layer_tensors[ln], src.chunks[0].layer_tensors[ln])
    conn.request_finished(dst)
```

- [ ] **Step 2: Create `docs/phase3-gpu-box.md`**

```markdown
# Phase 3 — building & running BaM runtime on the GPU box

Prereqs (one-time): IOMMU disabled, Above-4G Decoding + Resizable BAR on
(`nvidia-smi -q -d MEMORY | grep -iA3 BAR1` should show GBs), BaM built so that
`$BAM_HOME/build/lib/libnvm.so` and `$BAM_HOME/include` exist, BaM kernel module
loaded so `/dev/libnvm0` exists, and `/dev/nvme0n1` is a wipeable spare bound to BaM.

## Build the native extension
```bash
export BAM_HOME=/path/to/bam
export BAM_KV_BUILD_NATIVE=1
export BAM_KV_CUDA_ARCH=89          # RTX 6000 Ada
pip install -e ".[test,native]"
python -c "import bam_kv_cache._native; print('native OK')"
```

## Run the hardware round-trip
```bash
export BAM_KV_RUN_HW=1
export BAM_KV_NVME=/dev/libnvm0
python -m pytest tests/test_native_roundtrip.py -v
```

The pure-Python suite (`python -m pytest -q`, 47 tests) runs with or without the
native build. `tests/test_native_roundtrip.py` is skipped unless
`bam_kv_cache._native` is importable, CUDA is available, and `BAM_KV_RUN_HW=1`.
```

- [ ] **Step 3: Verify the round-trip is skipped locally (no CUDA / no native)**

Run: `python3 -m pytest tests/test_native_roundtrip.py -v`
Expected: skipped (1 skipped) — `bam_kv_cache._native` not importable locally.

- [ ] **Step 4: Run full local suite**

Run: `python3 -m pytest -q`
Expected: all pass, 1 skipped (47 passed, 1 skipped).

- [ ] **Step 5: Commit**

```bash
git add tests/test_native_roundtrip.py docs/phase3-gpu-box.md
git commit -m "test(native): GPU-box BaM round-trip (skipped off-box) + build docs"
```

---

## Self-review notes (already applied)

- **Spec coverage:** `BamDeviceConfig` (§4.1) → Task 1; `SsdAllocator`/metadata widen → Task 2; `RawNamespaceAllocator` (§4.4) → Task 3; native header/impl/bindings (§4.2) → Tasks 4–6; build (§6) → Task 7; `NativeBamRuntime` (§4.3) → Task 8; wiring (§5) + testing (§7) → Task 9. Page-cache→D2D data path is in Task 5's store/load.
- **Type consistency:** `BamDeviceConfig` field names match between Task 1, the pybind ctor (Task 6), and `from_config` (Task 8). The native `submit_store/submit_load(dptr, nbytes, dev_offset, stream)` signature matches across header (Task 4), `.cu` (Task 5), bindings (Task 6), and `NativeBamRuntime` (Task 8). `IoStatusCode {0,1,2}` (Task 4/5) maps to `IoStatus.PENDING/DONE/FAILED` in `_STATUS` (Task 8). `KVTensorDescriptor(tensor, layer_id, nbytes)`, `SlotMapping(start, length)`, `SsdLocation(object_id, offset, nbytes)`, `IoHandle(handle_id, status, error)` reused from Phase 1–2 unchanged.
- **No connector/metadata logic change:** only `InProcessMetadataStore.__init__`'s type hint widens (Task 2); behavior identical, existing 34 tests stay green.
- **Known v1 simplifications (documented in spec §10):** page-region ring allocator serializes via a mutex (correct but limits overlap); one contiguous extent per (chunk,layer); page-cache→D2D copy rather than true zero-copy DMA. The `.cu` BaM API calls must be validated against `$BAM_HOME` headers on the box (Task 5 validation note).
```
