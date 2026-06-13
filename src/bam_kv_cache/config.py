from dataclasses import dataclass


@dataclass(frozen=True)
class KVConfig:
    """Static configuration that defines how KV chunks are keyed and stored.

    Every field that changes KV interpretation feeds the object key, so two
    runs with different configs never collide on the same SSD payload.
    """

    model_id: str
    tokenizer_id: str
    kv_dtype: str          # e.g. "float32", "float16", "bfloat16"
    kv_layout: str         # e.g. "NHD", "HND"
    chunk_size: int        # tokens per KV chunk
    tp_rank: int
    ssd_root: str          # directory backing the mock SSD object store
    block_alignment: int = 4096
