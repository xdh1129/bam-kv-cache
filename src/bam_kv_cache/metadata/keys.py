import hashlib
from typing import NewType, Optional

# An object key is the stable hex identity of one (chunk, layer, rank, ...) KV blob.
ObjectKey = NewType("ObjectKey", str)

_FIELD_SEP = "\x1f"  # ASCII unit separator; cannot appear in normal field text


def build_object_key(
    *,
    model_id: str,
    tokenizer_id: str,
    token_chunk_hash: str,
    layer_id: int,
    tp_rank: int,
    kv_dtype: str,
    kv_layout: str,
    chunk_size: int,
    hybrid_geometry: Optional[str] = None,
) -> ObjectKey:
    """Deterministically hash all fields that affect KV interpretation.

    `token_chunk_hash` is opaque to the connector: the caller (tokenizer /
    prefix hasher) produces it. `hybrid_geometry` is accepted for forward
    compatibility (Phase 6) but is empty for non-hybrid v0 models.
    """
    h = hashlib.blake2b(digest_size=16)
    parts = [
        model_id,
        tokenizer_id,
        token_chunk_hash,
        str(layer_id),
        str(tp_rank),
        kv_dtype,
        kv_layout,
        str(chunk_size),
        hybrid_geometry or "",
    ]
    h.update(_FIELD_SEP.join(parts).encode("utf-8"))
    return ObjectKey(h.hexdigest())
