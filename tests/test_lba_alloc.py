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
