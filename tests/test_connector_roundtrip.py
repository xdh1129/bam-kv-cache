import pytest
import torch

from bam_kv_cache.config import KVConfig
from bam_kv_cache.connector.bam_connector import (
    BamConnectorCore,
    BamConnectorError,
    KVChunk,
    KVRequest,
)
from bam_kv_cache.metadata.store import InProcessMetadataStore
from bam_kv_cache.runtime.mock import MockBamRuntime
from bam_kv_cache.ssd.object_store import FileObjectStore

LAYERS = ["layer.0", "layer.1"]


def _build(tmp_path):
    store = FileObjectStore(str(tmp_path))
    meta = InProcessMetadataStore(store)
    rt = MockBamRuntime(store)
    cfg = KVConfig(
        model_id="m", tokenizer_id="t", kv_dtype="float32", kv_layout="NHD",
        chunk_size=16, tp_rank=0, ssd_root=str(tmp_path),
    )
    conn = BamConnectorCore(cfg, meta, rt)
    conn.register_kv_caches({ln: None for ln in LAYERS})
    return conn, meta, rt, store


def _request_with_data(request_id, seed=0):
    # Two chunks, each with a distinct tensor per layer.
    chunks = []
    for ci in range(2):
        layer_tensors = {
            ln: torch.arange(8, dtype=torch.float32) + (seed + ci * 10 + li)
            for li, ln in enumerate(LAYERS)
        }
        chunks.append(KVChunk(chunk_hash=f"chunk-{ci}", layer_tensors=layer_tensors))
    return KVRequest(request_id=request_id, chunks=chunks)


def _store_request(conn, req):
    for ln in LAYERS:
        conn.save_kv_layer(ln, req)
    conn.wait_for_save()


def test_store_then_hit_and_load_roundtrip(tmp_path):
    conn, meta, rt, store = _build(tmp_path)

    src = _request_with_data("req-1")
    _store_request(conn, src)

    # Second request, same chunk hashes -> full hit on both chunks.
    dst = KVRequest(
        request_id="req-2",
        chunks=[
            KVChunk(chunk_hash=c.chunk_hash, layer_tensors={ln: torch.zeros(8) for ln in LAYERS})
            for c in src.chunks
        ],
    )
    matched = conn.get_num_new_matched_tokens(dst)
    assert matched == 2 * 16  # 2 chunks * chunk_size

    conn.start_load_kv(dst)
    for ln in LAYERS:
        conn.wait_for_layer_load(ln)

    # Loaded buffers must equal the originally stored tensors.
    for sc, dc in zip(src.chunks, dst.chunks):
        for ln in LAYERS:
            assert torch.equal(dc.layer_tensors[ln], sc.layer_tensors[ln])

    conn.request_finished(dst)
    # All refs released -> records evictable.
    assert len(meta.evict_ready(limit=10)) == 4  # 2 chunks * 2 layers


def test_miss_when_not_stored(tmp_path):
    conn, meta, rt, store = _build(tmp_path)
    dst = _request_with_data("req-cold")
    assert conn.get_num_new_matched_tokens(dst) == 0


def test_no_key_collision_across_layers(tmp_path):
    conn, meta, rt, store = _build(tmp_path)
    k0 = conn._key("chunk-0", "layer.0")
    k1 = conn._key("chunk-0", "layer.1")
    assert k0 != k1


def test_load_failure_marks_failed_and_excludes(tmp_path):
    conn, meta, rt, store = _build(tmp_path)
    src = _request_with_data("req-1")
    _store_request(conn, src)

    # Force every load to fail.
    for ln in LAYERS:
        for chunk in src.chunks:
            rec = meta.lookup_meta([conn._key(chunk.chunk_hash, ln)])
            for r in rec.values():
                rt.fail_on(r.ssd_object_id)

    dst = KVRequest(
        request_id="req-2",
        chunks=[
            KVChunk(chunk_hash=c.chunk_hash, layer_tensors={ln: torch.zeros(8) for ln in LAYERS})
            for c in src.chunks
        ],
    )
    conn.start_load_kv(dst)
    with pytest.raises(BamConnectorError):
        for ln in LAYERS:
            conn.wait_for_layer_load(ln)

    # The failed key is no longer a hit.
    failed_key = conn._key(src.chunks[0].chunk_hash, LAYERS[0])
    assert meta.lookup_meta([failed_key]) == {}
