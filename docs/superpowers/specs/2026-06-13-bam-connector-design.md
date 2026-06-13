# BaM KV-Cache Connector — Implementation Spec (Phases 1+2)

**Date:** 2026-06-13
**Scope:** Control plane (metadata) + `BamConnector` core with a mock BaM runtime.
**Parent design:** `../../../../bam-lmcache-hybrid-design.md` (BaM/LMCache hybrid architecture).

## 1. Goal

Implement the parts of the BaM-direct KV-cache architecture that are pure-Python
and fully unit-testable on a CPU-only machine (no GPU / no BaM hardware):

- **Phase 1** — metadata control plane: object-key generation, metadata record,
  state machine, and an in-process metadata API.
- **Phase 2** — `BamConnectorCore`: the vLLM connector hook logic, driving the
  metadata plane and a **mock** BaM runtime so fake KV tensors can round-trip
  GPU-buffer → SSD → GPU-buffer entirely on CPU.

The real CUDA/BaM data plane (Phase 3), the LMCache MP metadata service (Phase 4),
benchmarks (Phase 5), and HMA/Gemma4 hybrid support (Phase 6) are **out of scope**
here but are isolated behind interfaces so they can drop in without touching the
connector or metadata code.

## 2. Constraints / environment

- Dev machine: macOS, Python 3.12, torch 2.10 **CPU-only**, no NVIDIA GPU.
- The vendored `vllm/` does **not** import on this machine (missing `zmq`; `base.py`
  pulls in the full `vllm.distributed` + torch-distributed stack). Therefore the
  connector core **must not** import `vllm`. It is written against a thin local
  interface that mirrors the `KVConnectorBase_V1` hook signatures. A real subclass
  adapter is added later in the GPU environment (Risk 15.1 — isolate vLLM API
  instability).

## 3. Package layout

```
bam-kv-cache/
  pyproject.toml                  # pkg metadata + pytest config, no GPU deps
  src/bam_kv_cache/
    __init__.py
    config.py                     # KVConfig: model_id, tokenizer_id, dtype, layout, chunk_size, tp_rank, ssd_root
    metadata/
      __init__.py
      keys.py                     # build_object_key(...) -> ObjectKey
      record.py                   # State (enum), MetadataRecord (dataclass)
      state_machine.py            # ALLOWED_TRANSITIONS + validate_transition()
      api.py                      # MetadataPlane (ABC) — LMCache-MP swap point (Phase 4)
      store.py                    # InProcessMetadataStore(MetadataPlane)
    runtime/
      __init__.py
      descriptors.py              # KVTensorDescriptor, SlotMapping, SsdLocation, IoHandle, IoStatus
      base.py                     # BamRuntime (ABC) — CUDA swap point (Phase 3)
      mock.py                     # MockBamRuntime(BamRuntime) — CPU, file-backed SSD
    ssd/
      __init__.py
      object_store.py             # FileObjectStore: 4KiB-aligned append/extent alloc
    connector/
      __init__.py
      vllm_interface.py           # KVConnectorLike (ABC) mirroring vLLM hook signatures, NO vllm import
      bam_connector.py            # BamConnectorCore(KVConnectorLike)
  tests/
    test_keys.py
    test_state_machine.py
    test_metadata_store.py
    test_object_store.py
    test_mock_runtime.py
    test_connector_roundtrip.py
```

## 4. Interfaces (the two hard boundaries)

### 4.1 `MetadataPlane` (ABC) — `metadata/api.py`

In-process now; LMCache MP/ZMQ metadata-only service later. Method names mirror
the design doc API (§8):

```python
lookup_meta(keys)            -> dict[ObjectKey, MetadataRecord]   # ready records only
reserve_store(keys, sizes)   -> dict[ObjectKey, SsdLocation]      # state -> writing
commit_store(keys)           -> None                              # state -> ready
abort_store(keys)            -> None                              # state -> aborted
begin_load(keys)             -> dict[ObjectKey, MetadataRecord]   # ref_count+=1, state -> loading
release(keys)                -> None                              # ref_count-=1, -> ready if 0
mark_failed(keys, reason)    -> None                              # state -> failed
pin(keys) / unpin(keys)      -> None
evict_ready(limit, policy)   -> list[ObjectKey]                   # ready && ref_count==0 && !pinned
stats()                      -> MetadataStats
```

Rules: `lookup_meta` returns only `ready`; `reserve`/`commit` are two-phase so a
half-written payload is never a hit; `loading` holds `ref_count > 0`;
`ref_count > 0` blocks evict; `failed` records are never returned by lookup.

### 4.2 `BamRuntime` (ABC) — `runtime/base.py`

Mock now; CUDA/C++ extension later. Never touches cache policy or prompt semantics.

```python
submit_store(tensor: KVTensorDescriptor, slot_mapping: SlotMapping,
             location: SsdLocation, stream=None) -> IoHandle
submit_load(tensor: KVTensorDescriptor, slot_mapping: SlotMapping,
            location: SsdLocation, stream=None) -> IoHandle
poll(handle: IoHandle) -> IoStatus
wait(handle: IoHandle) -> IoStatus
```

`MockBamRuntime` backs `SsdLocation` with a `FileObjectStore` directory and copies
real torch **CPU** tensors. It must never call `.cpu()` semantics that the real
runtime forbids — the mock simply reads/writes the tensor bytes it is given.

## 5. Connector core — `connector/bam_connector.py`

`BamConnectorCore` depends only on `MetadataPlane` + `BamRuntime` + a registered
KV-buffer description. Implements the hook subset (mirroring vLLM):

```
register_kv_caches(kv_caches)          # build per-layer KVTensorDescriptor table
get_num_new_matched_tokens(request)    # candidate keys -> lookup_meta -> matched token count
start_load_kv(request)                 # begin_load -> submit_load per (layer, chunk)
wait_for_layer_load(layer_name)        # wait(handle) -> release  (layer-aware)
save_kv_layer(layer_name, request)     # build key -> reserve_store -> submit_store
wait_for_save()                        # wait(handle) -> commit_store / abort / mark_failed
request_finished(request)              # release any held refs, cleanup
```

### Data flow — store (write-through, two-phase)
`save_kv_layer` → `build_object_key` → `reserve_store` → `submit_store(tensor_range, ssd_loc)`
→ `wait_for_save` → `commit_store` on success / `abort_store` or `mark_failed` on failure.

### Data flow — load (layer-aware, fail-fast)
`get_num_new_matched_tokens` → candidate keys → `lookup_meta` (ready only) → report matched
tokens. `start_load_kv` → `begin_load` (ref++) → `submit_load`. `wait_for_layer_load` →
`wait` → `release`. On load failure → `mark_failed`; future lookups exclude it (no recompute
fallback in v1, per design §11).

## 6. Object key — `metadata/keys.py`

```
object_key = hash(model_id, tokenizer_id, token_chunk_hash, layer_id, tp_rank,
                  kv_dtype, kv_layout, chunk_size, optional_hybrid_geometry)
```
Stable, collision-resistant (e.g. blake2b of a canonical field encoding).
`optional_hybrid_geometry` is accepted but unused in v0 (non-hybrid only).

## 7. Testing (implement-then-test, pytest, CPU-only)

- `test_keys`: determinism; any field change changes the key; no collision across
  layer / tp_rank / chunk.
- `test_state_machine`: every legal transition allowed; illegal transitions raise;
  `failed`/`aborted`/`reserved`/`writing` never lookup-visible.
- `test_metadata_store`: reserve→commit makes a key visible; `begin_load` bumps
  ref_count and blocks evict; `release` restores; `mark_failed` excludes from lookup.
- `test_object_store`: 4KiB-aligned alloc, byte round-trip, distinct offsets.
- `test_mock_runtime`: `submit_store`/`submit_load` round-trips a CPU tensor through
  the file store; `poll`/`wait` report completion; injected failure surfaces in status.
- `test_connector_roundtrip`: fake KV tensors store→commit, then a second request
  hits lookup and loads back identical bytes; a forced load failure marks `failed`
  and is excluded from the next lookup.

## 8. Completion criteria (from design §14, Phases 1–2)

- Store/load state transitions correct; `failed`/`aborted` never hit on lookup;
  `ref_count` blocks eviction.
- Fake KV tensor round-trips through the connector + mock runtime.
- Keys do not collide across layer / rank / chunk.
- Load failure marks `failed`.
- No `vllm` import in `bam_kv_cache`; connector core importable and tested on CPU.

## 9. Deferred (behind interfaces)

- Phase 3: CUDA/C++ extension implementing `BamRuntime` (GPU-direct, no staging).
- Phase 4: LMCache MP/ZMQ implementation of `MetadataPlane`.
- Phase 5: benchmark harness (CPU/DRAM vs POSIX vs GDS vs BaM).
- Phase 6: `SupportsHMA`, per-layer geometry, sliding-window valid range, Gemma4.
- vLLM adapter subclassing the real `KVConnectorBase_V1`, delegating to
  `BamConnectorCore` (built/run only in the GPU environment).
