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
