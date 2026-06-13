# bam-kv-cache

BaM-direct KV cache connector for LMCache/vLLM. Splits the cache into a
**control plane** (metadata: existence, state, ref counts, eviction) and a
**data plane** (BaM moves KV payload between GPU memory and SSD).

This repo currently implements **Phases 1–2** — the metadata control plane and a
`BamConnectorCore` that round-trips KV tensors through a **mock** BaM runtime,
fully on CPU. Real CUDA/BaM (Phase 3), the LMCache MP metadata service (Phase 4),
and HMA/Gemma4 support (Phase 6) are isolated behind the `BamRuntime` and
`MetadataPlane` interfaces.

## Layout

- `src/bam_kv_cache/metadata/` — object keys, records, state machine, in-process plane
- `src/bam_kv_cache/runtime/` — `BamRuntime` ABC + CPU mock
- `src/bam_kv_cache/ssd/` — file-backed object store
- `src/bam_kv_cache/connector/` — `BamConnectorCore` (vLLM hook logic, no vllm import)

## Test

```bash
pip install -e ".[test]"
python -m pytest -v
```

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for design + plan.
