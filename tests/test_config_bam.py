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
