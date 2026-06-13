from bam_kv_cache.metadata.keys import build_object_key


def _base(**overrides):
    kwargs = dict(
        model_id="llama-3",
        tokenizer_id="tok-1",
        token_chunk_hash="abc",
        layer_id=0,
        tp_rank=0,
        kv_dtype="bfloat16",
        kv_layout="NHD",
        chunk_size=16,
    )
    kwargs.update(overrides)
    return build_object_key(**kwargs)


def test_deterministic():
    assert _base() == _base()


def test_layer_changes_key():
    assert _base(layer_id=0) != _base(layer_id=1)


def test_tp_rank_changes_key():
    assert _base(tp_rank=0) != _base(tp_rank=1)


def test_chunk_hash_changes_key():
    assert _base(token_chunk_hash="abc") != _base(token_chunk_hash="abd")


def test_no_collision_across_fields():
    # A grid of (layer, rank, chunk) must yield all-distinct keys.
    keys = {
        _base(layer_id=l, tp_rank=r, token_chunk_hash=c)
        for l in range(3)
        for r in range(2)
        for c in ("c0", "c1", "c2")
    }
    assert len(keys) == 3 * 2 * 3


def test_separator_avoids_field_smearing():
    # "a","bc" must differ from "ab","c" — separator prevents concatenation collisions.
    k1 = _base(model_id="a", tokenizer_id="bc")
    k2 = _base(model_id="ab", tokenizer_id="c")
    assert k1 != k2
