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
