# BaM Runtime (Phase 3) — Implementation Spec

**Date:** 2026-06-13
**Scope:** Real BaM CUDA implementation of the `BamRuntime` data plane.
**Depends on:** Phases 1–2 (`docs/superpowers/specs/2026-06-13-bam-connector-design.md`).
**Parent design:** `../../../../bam-lmcache-hybrid-design.md` §6.3, §10, §14 (Phase 3).

## 1. Goal

Provide a concrete `BamRuntime` that moves KV payload directly between GPU memory
and an NVMe SSD using BaM (GPU-initiated storage access), replacing the CPU
`MockBamRuntime` without touching `BamConnectorCore` or the metadata plane.

The connector already depends only on the `BamRuntime` ABC and `SsdLocation`
(byte `offset` + `nbytes`). Phase 3 adds:

1. A native CUDA extension (`BamDevice`) that owns BaM `Controller`s + `page_cache`
   and performs store/load by LBA into/out of a caller-provided GPU buffer.
2. A Python `NativeBamRuntime(BamRuntime)` that marshals `KVTensorDescriptor` /
   `SsdLocation` / CUDA stream into the native calls and maps completion to
   `IoStatus`.
3. A `RawNamespaceAllocator` that bump-allocates LBA-aligned byte extents on the
   raw namespace, satisfying the same allocator contract the metadata plane uses.

## 2. Target environment (GPU box only)

This code compiles and runs **only** on the BaM-capable machine. There is **no
local (Mac) build or test** for the native path — per decision, all verification
happens on the GPU box.

Requirements (from BaM README): x86 with PCIe P2P, Volta+ GPU exposing its memory
over a large PCIe BAR1 window, a dedicated NVMe SSD bound to BaM's kernel module,
IOMMU disabled + Above-4G Decoding, CUDA 12.3+, BaM built (its CMake produces
`libnvm`). The NVMe namespace is **raw block space** — there is no filesystem; the
device may be overwritten.

**Confirmed target box** (`aiserver-nv-6000ada-x8-122`): 7× RTX 6000 Ada
(~48 GB, **compute 8.9 → sm_89**), **BAR1 = 65536 MiB (ReBAR on, P2P viable)**.
NVMe `/dev/nvme0n1` ≈ 1.86 TiB (`2048408248320` bytes), LBA format 0 = **512 B**
in use (format 1 = 4096 B available; reformatting to 4096 to match `page_size` is
optional and would wipe the drive). Pre-flight to confirm before running:
`cat /proc/cmdline | grep -i iommu` (IOMMU off) and that `/dev/nvme0n1` is a
wipeable spare, not the OS disk.

## 3. Architecture

```
        Python                              native extension (bam_kv_cache._native)
┌────────────────────────┐          ┌───────────────────────────────────────────┐
│ BamConnectorCore        │          │ bindings.cpp  (pybind11, torch ext)         │
│   (UNCHANGED)           │          │   class BamDevice:                          │
│        │ BamRuntime ABC │   FFI    │     submit_store(dptr,nbytes,off,stream)->h │
│        ▼                │ ───────▶ │     submit_load (dptr,nbytes,off,stream)->h │
│ NativeBamRuntime        │          │     poll(h)->int ; wait(h)->int             │
│        │                │          │            │                                │
│        ▼ allocate()     │          │   bam_runtime.cu  (CUDA + BaM)              │
│ RawNamespaceAllocator   │          │     Controller[], page_cache_t,             │
└────────────────────────┘          │     read/write kernels, cudaEvent table     │
        ▲                            └───────────────────────────────────────────┘
        │ SsdLocation(offset,nbytes)            │ nvm_cmd_rw_blks(lba, nblocks)
   InProcessMetadataStore                        ▼
   (control plane, UNCHANGED logic)          raw NVMe namespace
```

Control/data split is preserved: the metadata plane decides *where* (allocates an
LBA extent, tracks state); BaM moves the *bytes* GPU↔SSD.

## 4. Components

### 4.1 `config_bam.py` — `BamDeviceConfig`

```python
@dataclass(frozen=True)
class BamDeviceConfig:
    nvme_paths: tuple[str, ...]      # BaM controller device paths, e.g. ("/dev/libnvm0",)
    nvm_namespace: int               # NVMe namespace id (default 1)
    cuda_device: int                 # CUDA device ordinal
    queue_depth: int                 # per-queue depth
    num_queues: int                  # NVMe submission/completion queue pairs
    page_size: int                   # BaM page cache page size (bytes)
    page_cache_pages: int            # number of pages in the GPU page cache
    lba_block_size: int              # NVMe LBA format size (e.g. 512 or 4096)
    namespace_capacity_bytes: int    # usable raw capacity for the allocator
```

### 4.2 `native/` — CUDA extension

- `bam_runtime.h`: opaque `BamDevice` declaration with C++-only types (no CUDA
  types in the header) so `bindings.cpp` stays CUDA-free.
- `bam_runtime.cu`: implements `BamDevice`:
  - **ctor**: open one `Controller` per `nvme_paths` entry (mirrors
    `benchmarks/readwrite/main.cu` construction), build a `page_cache_t` on
    `cuda_device` sized `page_cache_pages * page_size`, allocate a `cudaEvent_t`
    table for handles.
  - **`submit_store(dptr, nbytes, dev_offset, stream)`**: copy `nbytes` from the
    caller's GPU buffer at `dptr` into the page-cache GPU buffer
    (`cudaMemcpyAsync D2D` on `stream`), launch the BaM write kernel issuing
    `nvm_cmd_rw_blks(NVM_IO_WRITE, lba, nblocks)` where `lba = dev_offset /
    lba_block_size`, `nblocks = ceil(nbytes / lba_block_size)`; record a
    `cudaEvent` on `stream`; return a handle id.
  - **`submit_load(dptr, nbytes, dev_offset, stream)`**: launch the BaM read
    kernel (`NVM_IO_READ`) filling the page-cache buffer, then
    `cudaMemcpyAsync D2D` page-cache → `dptr`; record a `cudaEvent`; return a
    handle id.
  - **`poll(h)`**: `cudaEventQuery` → `0=PENDING, 1=DONE`; on CUDA/NVM error
    `2=FAILED`.
  - **`wait(h)`**: `cudaEventSynchronize` → `1=DONE` / `2=FAILED`.
- `bindings.cpp`: pybind11 wrapper exposing `BamDevice(BamDeviceConfig fields)`
  and the four methods. Built as a torch extension; module name
  `bam_kv_cache._native`.

**v1 data path (chosen):** BaM `page_cache` GPU buffer + one `cudaMemcpyAsync`
DeviceToDevice into/out of the KV tensor. No CPU staging. True zero-extra-copy
(NVMe DMA straight into the KV tensor address) is deferred.

### 4.3 `runtime/native_bam.py` — `NativeBamRuntime(BamRuntime)`

```python
class NativeBamRuntime(BamRuntime):
    def __init__(self, device):           # device: bam_kv_cache._native.BamDevice
        self._dev = device

    def submit_store(self, kv, slot_mapping, location, stream=None) -> IoHandle:
        sp = _stream_ptr(stream)           # torch.cuda.current_stream().cuda_stream if None
        hid = self._dev.submit_store(kv.tensor.data_ptr(), kv.nbytes, location.offset, sp)
        return IoHandle(handle_id=hid)

    def submit_load(self, kv, slot_mapping, location, stream=None) -> IoHandle:
        sp = _stream_ptr(stream)
        hid = self._dev.submit_load(kv.tensor.data_ptr(), kv.nbytes, location.offset, sp)
        return IoHandle(handle_id=hid)

    def poll(self, handle): return _to_status(self._dev.poll(handle.handle_id), handle)
    def wait(self, handle): return _to_status(self._dev.wait(handle.handle_id), handle)
```

`_to_status` maps `0/1/2 -> IoStatus.PENDING/DONE/FAILED` and sets `handle.status`
(and `handle.error` on failure), mirroring `MockBamRuntime` semantics so the
connector's fail-fast path is unchanged.

### 4.4 `runtime/lba_alloc.py` — `RawNamespaceAllocator`

Same `allocate(nbytes) -> SsdLocation` contract as `FileObjectStore`, but over the
raw namespace (no file). Bump allocator aligned to `max(lba_block_size,
alignment)`; raises `RuntimeError` when `namespace_capacity_bytes` is exceeded.
`object_id` is a monotonic counter; `offset` is a byte offset that is a multiple
of `lba_block_size`.

The metadata plane already calls `self._store.allocate(nbytes)`. Phase 3 widens
that dependency to an `SsdAllocator` Protocol (`allocate(nbytes) -> SsdLocation`)
that both `FileObjectStore` and `RawNamespaceAllocator` satisfy — a one-line
type-hint change in `InProcessMetadataStore`, no behavior change.

## 5. Wiring (on the GPU box)

```python
dev_cfg = BamDeviceConfig(nvme_paths=("/dev/libnvm0",), nvm_namespace=1,
                          cuda_device=0, queue_depth=1024, num_queues=128,
                          page_size=4096, page_cache_pages=65536,   # 256 MiB cache; size to workload
                          lba_block_size=512,                        # in use on /dev/nvme0n1
                          namespace_capacity_bytes=2048408248320)    # ~1.86 TiB
device   = bam_kv_cache._native.BamDevice(**asdict(dev_cfg))
runtime  = NativeBamRuntime(device)
alloc    = RawNamespaceAllocator(dev_cfg.namespace_capacity_bytes, dev_cfg.lba_block_size)
meta     = InProcessMetadataStore(alloc)
conn     = BamConnectorCore(kv_cfg, meta, runtime)   # KV tensors must be CUDA tensors
```

## 6. Build

`setup.py` defines the extension only when `BAM_KV_BUILD_NATIVE=1`:

```python
# pseudo
if os.environ.get("BAM_KV_BUILD_NATIVE") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    bam_home = os.environ["BAM_HOME"]            # path to built BaM repo
    ext = CUDAExtension(
        name="bam_kv_cache._native",
        sources=["native/bindings.cpp", "native/bam_runtime.cu"],
        include_dirs=[f"{bam_home}/include"],
        library_dirs=[f"{bam_home}/build/lib"],
        libraries=["nvm"],
        extra_compile_args={"nvcc": ["-O3", "-std=c++17",
                                     "-gencode=arch=compute_89,code=sm_89"]},  # RTX 6000 Ada
    )
```

`pip install -e ".[native]"` with `BAM_KV_BUILD_NATIVE=1 BAM_HOME=/path/to/bam`
builds it. Without the env var (local dev), the package installs pure-Python and
`bam_kv_cache._native` is simply absent.

## 7. Testing (GPU box only)

A pytest module `tests/test_native_roundtrip.py` guarded by
`pytest.importorskip("bam_kv_cache._native")` and a CUDA check, so it is skipped
everywhere except the GPU box:

- **round-trip:** build the full stack (§5); store a CUDA KV tensor, then load it
  into a zeroed CUDA tensor of the same shape; assert `torch.equal` after `wait`.
- **multi-chunk / multi-layer:** distinct `(chunk, layer)` extents do not overlap
  and load back independently.
- **async ordering:** `poll` returns `PENDING` before `wait`, `DONE` after, on a
  non-default stream.
- **failure surfaces:** an out-of-range LBA (or forced NVM error) yields
  `IoStatus.FAILED` with a non-None error, and the connector marks the record
  failed (fail-fast).

No local CPU test for the native path; the existing 34 Phase 1–2 tests continue
to cover the connector/metadata/mock paths.

## 8. Error handling

NVM/CUDA errors inside `submit_*`/`poll`/`wait` are caught natively and surfaced
as status `2=FAILED` with a message; `NativeBamRuntime` turns that into
`IoStatus.FAILED` + `handle.error`. The connector's existing `wait_for_layer_load`
fail-fast (`mark_failed` + raise) handles the rest unchanged.

## 9. Risks

- **Page-cache sizing:** a KV chunk-layer tensor must fit the page-cache transfer
  granularity; size `page_cache_pages * page_size` to cover the largest chunk and
  enough concurrent in-flight transfers.
- **LBA alignment / capacity:** `offset` must be `lba_block_size`-aligned (the
  allocator guarantees it); `nblocks` rounds up, so the last block may be partly
  unused — acceptable for v1.
- **Stream correctness:** the D2D copy and the BaM kernel must be ordered on the
  same `stream`; the recorded event must follow both.
- **Concurrent handles:** the native `cudaEvent` table must be thread-safe and
  bounded (recycle completed events).
- **Header hygiene:** keep CUDA/BaM types out of `bam_runtime.h` so `bindings.cpp`
  compiles without CUDA-only includes.

## 10. Out of scope (deferred)

- True zero-extra-copy (NVMe DMA directly into the KV tensor address).
- GDS/cuFile and POSIX baselines (Phase 5 benchmark).
- LMCache MP metadata service (Phase 4).
- HMA/Gemma4 hybrid geometry (Phase 6).
- Paged per-block `SlotMapping` (v1 uses one contiguous extent per chunk-layer).
- Tensor-parallel multi-GPU sharding specifics beyond `tp_rank` already in the key.
