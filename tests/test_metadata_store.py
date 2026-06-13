import pytest

from bam_kv_cache.metadata.keys import build_object_key
from bam_kv_cache.metadata.record import State
from bam_kv_cache.metadata.store import InProcessMetadataStore
from bam_kv_cache.ssd.object_store import FileObjectStore


@pytest.fixture
def meta(tmp_path):
    return InProcessMetadataStore(FileObjectStore(str(tmp_path)))


def _key(i):
    return build_object_key(
        model_id="m", tokenizer_id="t", token_chunk_hash=f"c{i}",
        layer_id=0, tp_rank=0, kv_dtype="float32", kv_layout="NHD", chunk_size=16,
    )


def test_reserved_is_not_a_hit(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    assert meta.lookup_meta([k]) == {}  # WRITING is not lookup-visible


def test_commit_makes_visible(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    meta.commit_store([k])
    assert k in meta.lookup_meta([k])


def test_abort_excludes_from_lookup(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    meta.abort_store([k])
    assert meta.lookup_meta([k]) == {}


def test_begin_load_bumps_ref_and_blocks_evict(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    meta.commit_store([k])
    recs = meta.begin_load([k])
    assert recs[k].ref_count == 1
    assert recs[k].state is State.LOADING
    assert meta.evict_ready(limit=10) == []  # ref_count > 0 cannot evict


def test_release_returns_to_ready_then_evictable(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    meta.commit_store([k])
    meta.begin_load([k])
    meta.release([k])
    assert meta.evict_ready(limit=10) == [k]


def test_mark_failed_excludes_and_clears_ref(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    meta.commit_store([k])
    meta.begin_load([k])
    meta.mark_failed([k], "boom")
    assert meta.lookup_meta([k]) == {}  # FAILED never a hit


def test_pin_blocks_evict(meta):
    k = _key(0)
    meta.reserve_store({k: 64})
    meta.commit_store([k])
    meta.pin([k])
    assert meta.evict_ready(limit=10) == []
    meta.unpin([k])
    assert meta.evict_ready(limit=10) == [k]


def test_reserve_returns_distinct_locations(meta):
    k0, k1 = _key(0), _key(1)
    locs = meta.reserve_store({k0: 64, k1: 64})
    assert locs[k0].offset != locs[k1].offset


def test_stats_counts(meta):
    k0, k1 = _key(0), _key(1)
    meta.reserve_store({k0: 64, k1: 64})
    meta.commit_store([k0])
    s = meta.stats()
    assert s.total == 2
    assert s.ready == 1
    assert s.writing == 1
