from bam_kv_cache import __version__
from bam_kv_cache.config import KVConfig


def test_package_imports():
    assert __version__ == "0.1.0"


def test_kvconfig_is_frozen():
    cfg = KVConfig(
        model_id="m",
        tokenizer_id="t",
        kv_dtype="float32",
        kv_layout="NHD",
        chunk_size=16,
        tp_rank=0,
        ssd_root="/tmp/bam",
    )
    assert cfg.chunk_size == 16
