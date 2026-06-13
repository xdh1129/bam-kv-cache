# BaM Connector (Phases 1+2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the BaM KV-cache control plane (metadata + state machine) and a `BamConnectorCore` that round-trips fake KV tensors GPU-buffer → SSD → GPU-buffer through a mock BaM runtime, fully on CPU.

**Architecture:** Two hard interface boundaries — `MetadataPlane` (in-process now, LMCache MP later) and `BamRuntime` (mock now, CUDA later). `BamConnectorCore` depends only on those two interfaces plus a thin `KVConnectorLike` ABC mirroring vLLM hook signatures, so it never imports `vllm` and stays testable on a CPU-only Mac.

**Tech Stack:** Python 3.12, torch 2.10 (CPU), pytest. No GPU, no CUDA, no `vllm` import.

**Spec:** `docs/superpowers/specs/2026-06-13-bam-connector-design.md`

**Convention for every task:** implement the module(s) first, then write tests, run them green, then commit (per the agreed implement-then-test approach). Run all pytest from the `bam-kv-cache/` repo root.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/bam_kv_cache/__init__.py`
- Create: `src/bam_kv_cache/config.py`
- Create: `src/bam_kv_cache/metadata/__init__.py`
- Create: `src/bam_kv_cache/runtime/__init__.py`
- Create: `src/bam_kv_cache/ssd/__init__.py`
- Create: `src/bam_kv_cache/connector/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "bam-kv-cache"
version = "0.1.0"
description = "BaM-direct KV cache connector (control plane + connector core)"
requires-python = ">=3.10"
dependencies = ["torch>=2.0"]

[project.optional-dependencies]
test = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create package `__init__.py` files**

`src/bam_kv_cache/__init__.py`:

```python
"""BaM-direct KV cache connector: control plane + connector core."""

__version__ = "0.1.0"
```

Create empty `__init__.py` for each subpackage: `metadata`, `runtime`, `ssd`, `connector`:

```python
```

- [ ] **Step 3: Create `src/bam_kv_cache/config.py`**

```python
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
```

- [ ] **Step 4: Create `tests/test_smoke.py`**

```python
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
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/bam_kv_cache tests/test_smoke.py
git commit -m "chore: scaffold bam_kv_cache package + KVConfig"
```

---

## Task 2: Object key generator

**Files:**
- Create: `src/bam_kv_cache/metadata/keys.py`
- Test: `tests/test_keys.py`

- [ ] **Step 1: Create `src/bam_kv_cache/metadata/keys.py`**

```python
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
```

- [ ] **Step 2: Create `tests/test_keys.py`**

```python
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
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_keys.py -v`
Expected: 6 passed.

- [ ] **Step 4: Commit**

```bash
git add src/bam_kv_cache/metadata/keys.py tests/test_keys.py
git commit -m "feat(metadata): object key generator (blake2b over keying fields)"
```

---

## Task 3: Metadata record + state machine

**Files:**
- Create: `src/bam_kv_cache/metadata/record.py`
- Create: `src/bam_kv_cache/metadata/state_machine.py`
- Test: `tests/test_state_machine.py`

**Note on states:** the design doc lists `reserved` and `writing` as separate states. In the v1 write-through connector, reservation and the start of the async write are the same event, so we collapse them into a single active `WRITING` state. This is an intentional, documented simplification.

- [ ] **Step 1: Create `src/bam_kv_cache/metadata/record.py`**

```python
import enum
import time
from dataclasses import dataclass, field

from .keys import ObjectKey


class State(enum.Enum):
    MISSING = "missing"
    WRITING = "writing"    # reserved + being written (collapsed, see plan note)
    READY = "ready"
    LOADING = "loading"
    ABORTED = "aborted"
    FAILED = "failed"
    EVICTED = "evicted"


@dataclass
class MetadataRecord:
    object_key: ObjectKey
    state: State
    ssd_object_id: int | None = None
    offset: int | None = None
    nbytes: int | None = None
    version: int = 0
    ref_count: int = 0
    pinned: bool = False
    last_access_ts: float = field(default_factory=time.monotonic)
    failure_reason: str | None = None
```

- [ ] **Step 2: Create `src/bam_kv_cache/metadata/state_machine.py`**

```python
from .record import State

# Legal forward transitions. Terminal states map to the empty set.
ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.MISSING: {State.WRITING},
    State.WRITING: {State.READY, State.ABORTED, State.FAILED},
    State.READY: {State.LOADING, State.FAILED, State.EVICTED},
    State.LOADING: {State.READY, State.FAILED},
    State.ABORTED: set(),
    State.FAILED: set(),
    State.EVICTED: set(),
}


class IllegalTransition(Exception):
    pass


def validate_transition(src: State, dst: State) -> None:
    if dst not in ALLOWED_TRANSITIONS[src]:
        raise IllegalTransition(f"{src.value} -> {dst.value} not allowed")
```

- [ ] **Step 3: Create `tests/test_state_machine.py`**

```python
import pytest

from bam_kv_cache.metadata.record import State
from bam_kv_cache.metadata.state_machine import (
    ALLOWED_TRANSITIONS,
    IllegalTransition,
    validate_transition,
)


def test_happy_store_path():
    validate_transition(State.MISSING, State.WRITING)
    validate_transition(State.WRITING, State.READY)


def test_happy_load_path():
    validate_transition(State.READY, State.LOADING)
    validate_transition(State.LOADING, State.READY)


def test_illegal_transition_raises():
    with pytest.raises(IllegalTransition):
        validate_transition(State.READY, State.WRITING)


def test_terminal_states_have_no_exits():
    for terminal in (State.ABORTED, State.FAILED, State.EVICTED):
        assert ALLOWED_TRANSITIONS[terminal] == set()


def test_writing_can_fail_or_abort():
    validate_transition(State.WRITING, State.FAILED)
    validate_transition(State.WRITING, State.ABORTED)


def test_loading_can_fail():
    validate_transition(State.LOADING, State.FAILED)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_state_machine.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bam_kv_cache/metadata/record.py src/bam_kv_cache/metadata/state_machine.py tests/test_state_machine.py
git commit -m "feat(metadata): MetadataRecord + state machine"
```

---

## Task 4: SSD object store (file-backed)

**Files:**
- Create: `src/bam_kv_cache/ssd/object_store.py`
- Test: `tests/test_object_store.py`

**Note:** `SsdLocation` lives here because it is an SSD addressing concept; other modules import it from `bam_kv_cache.ssd.object_store`.

- [ ] **Step 1: Create `src/bam_kv_cache/ssd/object_store.py`**

```python
import os
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SsdLocation:
    """Where a KV blob lives on the (mock) SSD."""

    object_id: int
    offset: int
    nbytes: int


class FileObjectStore:
    """File-backed stand-in for the SSD object store.

    Bump-allocates `block_alignment`-aligned extents inside a single append-only
    file. Thread-safe. Only stores raw bytes; validity is decided by the
    metadata plane, never here.
    """

    def __init__(self, root: str, block_alignment: int = 4096):
        os.makedirs(root, exist_ok=True)
        self._path = os.path.join(root, "objects.bin")
        self._alignment = block_alignment
        self._lock = threading.Lock()
        self._next_offset = 0
        self._next_object_id = 0
        open(self._path, "ab").close()  # ensure file exists

    def _align(self, n: int) -> int:
        a = self._alignment
        return ((n + a - 1) // a) * a

    def allocate(self, nbytes: int) -> SsdLocation:
        with self._lock:
            offset = self._next_offset
            object_id = self._next_object_id
            self._next_object_id += 1
            self._next_offset += self._align(nbytes)
            with open(self._path, "r+b") as f:
                f.seek(self._next_offset - 1)
                f.write(b"\x00")  # extend file to new high-water mark
        return SsdLocation(object_id=object_id, offset=offset, nbytes=nbytes)

    def write(self, loc: SsdLocation, data: bytes) -> None:
        if len(data) != loc.nbytes:
            raise ValueError(f"data size {len(data)} != location nbytes {loc.nbytes}")
        with self._lock:
            with open(self._path, "r+b") as f:
                f.seek(loc.offset)
                f.write(data)

    def read(self, loc: SsdLocation) -> bytes:
        with self._lock:
            with open(self._path, "rb") as f:
                f.seek(loc.offset)
                return f.read(loc.nbytes)
```

- [ ] **Step 2: Create `tests/test_object_store.py`**

```python
from bam_kv_cache.ssd.object_store import FileObjectStore, SsdLocation


def test_alloc_is_aligned(tmp_path):
    store = FileObjectStore(str(tmp_path), block_alignment=4096)
    a = store.allocate(100)
    b = store.allocate(100)
    assert a.offset == 0
    assert b.offset == 4096  # second alloc starts on next aligned block
    assert a.object_id != b.object_id


def test_write_read_roundtrip(tmp_path):
    store = FileObjectStore(str(tmp_path))
    loc = store.allocate(8)
    store.write(loc, b"12345678")
    assert store.read(loc) == b"12345678"


def test_distinct_extents_do_not_overlap(tmp_path):
    store = FileObjectStore(str(tmp_path))
    a = store.allocate(8)
    b = store.allocate(8)
    store.write(a, b"aaaaaaaa")
    store.write(b, b"bbbbbbbb")
    assert store.read(a) == b"aaaaaaaa"
    assert store.read(b) == b"bbbbbbbb"


def test_write_size_mismatch_raises(tmp_path):
    store = FileObjectStore(str(tmp_path))
    loc = store.allocate(8)
    try:
        store.write(loc, b"123")
    except ValueError:
        return
    raise AssertionError("expected ValueError on size mismatch")
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_object_store.py -v`
Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add src/bam_kv_cache/ssd/object_store.py tests/test_object_store.py
git commit -m "feat(ssd): file-backed object store with aligned extents"
```

---

## Task 5: BaM runtime — descriptors, ABC, mock

**Files:**
- Create: `src/bam_kv_cache/runtime/descriptors.py`
- Create: `src/bam_kv_cache/runtime/base.py`
- Create: `src/bam_kv_cache/runtime/mock.py`
- Test: `tests/test_mock_runtime.py`

- [ ] **Step 1: Create `src/bam_kv_cache/runtime/descriptors.py`**

```python
import enum
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVTensorDescriptor:
    """Describes a KV tensor region to move.

    In v1 `tensor` is a CPU torch.Tensor. The real runtime will instead carry a
    GPU pointer / paged-buffer descriptor; consumers only rely on the fields
    below, not on `tensor` being host memory.
    """

    tensor: Any        # torch.Tensor (CPU in mock)
    layer_id: int
    nbytes: int


@dataclass(frozen=True)
class SlotMapping:
    """Paged-KV slot mapping. v1 uses a single contiguous byte range."""

    start: int
    length: int


class IoStatus(enum.Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


@dataclass
class IoHandle:
    handle_id: int
    status: IoStatus = IoStatus.PENDING
    error: str | None = None
```

- [ ] **Step 2: Create `src/bam_kv_cache/runtime/base.py`**

```python
from abc import ABC, abstractmethod

from ..ssd.object_store import SsdLocation
from .descriptors import IoHandle, IoStatus, KVTensorDescriptor, SlotMapping


class BamRuntime(ABC):
    """Data plane: moves KV payload between GPU memory and SSD.

    Knows nothing about cache policy or prompt semantics. The mock implements
    this on CPU; a CUDA/C++ extension will implement the same interface later.
    """

    @abstractmethod
    def submit_store(
        self,
        kv: KVTensorDescriptor,
        slot_mapping: SlotMapping,
        location: SsdLocation,
        stream=None,
    ) -> IoHandle: ...

    @abstractmethod
    def submit_load(
        self,
        kv: KVTensorDescriptor,
        slot_mapping: SlotMapping,
        location: SsdLocation,
        stream=None,
    ) -> IoHandle: ...

    @abstractmethod
    def poll(self, handle: IoHandle) -> IoStatus: ...

    @abstractmethod
    def wait(self, handle: IoHandle) -> IoStatus: ...
```

- [ ] **Step 3: Create `src/bam_kv_cache/runtime/mock.py`**

```python
import itertools

import torch

from ..ssd.object_store import FileObjectStore, SsdLocation
from .base import BamRuntime
from .descriptors import IoHandle, IoStatus, KVTensorDescriptor, SlotMapping


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Dtype-agnostic byte view (works for float32/float16/bfloat16)."""
    flat = t.detach().contiguous().reshape(-1)
    return bytes(flat.view(torch.uint8).numpy().tobytes())


def _bytes_into_tensor(data: bytes, dst: torch.Tensor) -> None:
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    view = raw.view(dst.dtype).reshape(dst.shape)
    dst.copy_(view)


class MockBamRuntime(BamRuntime):
    """CPU implementation of the BaM data plane backed by a FileObjectStore.

    Performs I/O synchronously inside submit_* and reports completion via the
    handle. `fail_on(object_id)` injects a failure for testing the fail-fast
    path.
    """

    def __init__(self, object_store: FileObjectStore):
        self._store = object_store
        self._ids = itertools.count()
        self._fail_object_ids: set[int] = set()

    def fail_on(self, object_id: int) -> None:
        self._fail_object_ids.add(object_id)

    def submit_store(self, kv, slot_mapping, location, stream=None) -> IoHandle:
        h = IoHandle(handle_id=next(self._ids))
        try:
            if location.object_id in self._fail_object_ids:
                raise IOError("injected store failure")
            self._store.write(location, _tensor_to_bytes(kv.tensor))
            h.status = IoStatus.DONE
        except Exception as e:  # noqa: BLE001 - surface any failure as IoStatus.FAILED
            h.status = IoStatus.FAILED
            h.error = str(e)
        return h

    def submit_load(self, kv, slot_mapping, location, stream=None) -> IoHandle:
        h = IoHandle(handle_id=next(self._ids))
        try:
            if location.object_id in self._fail_object_ids:
                raise IOError("injected load failure")
            data = self._store.read(location)
            _bytes_into_tensor(data, kv.tensor)
            h.status = IoStatus.DONE
        except Exception as e:  # noqa: BLE001
            h.status = IoStatus.FAILED
            h.error = str(e)
        return h

    def poll(self, handle: IoHandle) -> IoStatus:
        return handle.status

    def wait(self, handle: IoHandle) -> IoStatus:
        return handle.status
```

- [ ] **Step 4: Create `tests/test_mock_runtime.py`**

```python
import torch

from bam_kv_cache.runtime.descriptors import IoStatus, KVTensorDescriptor, SlotMapping
from bam_kv_cache.runtime.mock import MockBamRuntime
from bam_kv_cache.ssd.object_store import FileObjectStore


def _desc(t: torch.Tensor) -> KVTensorDescriptor:
    return KVTensorDescriptor(tensor=t, layer_id=0, nbytes=t.element_size() * t.nelement())


def test_store_then_load_roundtrip(tmp_path):
    store = FileObjectStore(str(tmp_path))
    rt = MockBamRuntime(store)
    src = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    loc = store.allocate(_desc(src).nbytes)

    hs = rt.submit_store(_desc(src), SlotMapping(0, _desc(src).nbytes), loc)
    assert rt.wait(hs) is IoStatus.DONE

    dst = torch.zeros_like(src)
    hl = rt.submit_load(_desc(dst), SlotMapping(0, _desc(dst).nbytes), loc)
    assert rt.wait(hl) is IoStatus.DONE
    assert torch.equal(dst, src)


def test_injected_store_failure(tmp_path):
    store = FileObjectStore(str(tmp_path))
    rt = MockBamRuntime(store)
    src = torch.ones(4, dtype=torch.float32)
    loc = store.allocate(_desc(src).nbytes)
    rt.fail_on(loc.object_id)

    h = rt.submit_store(_desc(src), SlotMapping(0, _desc(src).nbytes), loc)
    assert rt.poll(h) is IoStatus.FAILED
    assert h.error is not None


def test_roundtrip_float16(tmp_path):
    store = FileObjectStore(str(tmp_path))
    rt = MockBamRuntime(store)
    src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    loc = store.allocate(_desc(src).nbytes)
    rt.submit_store(_desc(src), SlotMapping(0, _desc(src).nbytes), loc)
    dst = torch.zeros_like(src)
    rt.submit_load(_desc(dst), SlotMapping(0, _desc(dst).nbytes), loc)
    assert torch.equal(dst, src)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_mock_runtime.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/bam_kv_cache/runtime tests/test_mock_runtime.py
git commit -m "feat(runtime): BaM runtime ABC + descriptors + CPU mock"
```

---

## Task 6: Metadata plane (API + in-process store)

**Files:**
- Create: `src/bam_kv_cache/metadata/api.py`
- Create: `src/bam_kv_cache/metadata/store.py`
- Test: `tests/test_metadata_store.py`

- [ ] **Step 1: Create `src/bam_kv_cache/metadata/api.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..ssd.object_store import SsdLocation
from .keys import ObjectKey
from .record import MetadataRecord


@dataclass
class MetadataStats:
    total: int
    ready: int
    writing: int
    loading: int
    failed: int
    evicted: int
    hits: int
    misses: int


class MetadataPlane(ABC):
    """Control plane: owns key existence, state, ref counts, eviction safety.

    Never receives KV payload. In-process now; an LMCache MP/ZMQ metadata-only
    service implements the same interface later (Phase 4).
    """

    @abstractmethod
    def lookup_meta(self, keys: list[ObjectKey]) -> dict[ObjectKey, MetadataRecord]:
        """Return only READY records (a hit)."""

    @abstractmethod
    def reserve_store(self, sizes: dict[ObjectKey, int]) -> dict[ObjectKey, SsdLocation]:
        """Allocate SSD locations and mark keys WRITING (not yet hit-able)."""

    @abstractmethod
    def commit_store(self, keys: list[ObjectKey]) -> None:
        """WRITING -> READY."""

    @abstractmethod
    def abort_store(self, keys: list[ObjectKey]) -> None:
        """WRITING -> ABORTED."""

    @abstractmethod
    def begin_load(self, keys: list[ObjectKey]) -> dict[ObjectKey, MetadataRecord]:
        """ref_count += 1; READY -> LOADING. Returns the records (with locations)."""

    @abstractmethod
    def release(self, keys: list[ObjectKey]) -> None:
        """ref_count -= 1; LOADING -> READY when it reaches 0."""

    @abstractmethod
    def mark_failed(self, keys: list[ObjectKey], reason: str) -> None:
        """-> FAILED and ref_count = 0; never lookup-visible again."""

    @abstractmethod
    def pin(self, keys: list[ObjectKey]) -> None: ...

    @abstractmethod
    def unpin(self, keys: list[ObjectKey]) -> None: ...

    @abstractmethod
    def evict_ready(self, limit: int) -> list[ObjectKey]:
        """Evict up to `limit` records that are READY && ref_count==0 && !pinned."""

    @abstractmethod
    def stats(self) -> MetadataStats: ...
```

- [ ] **Step 2: Create `src/bam_kv_cache/metadata/store.py`**

```python
import threading
import time
from collections import Counter

from ..ssd.object_store import FileObjectStore, SsdLocation
from .api import MetadataPlane, MetadataStats
from .keys import ObjectKey
from .record import MetadataRecord, State
from .state_machine import validate_transition


class InProcessMetadataStore(MetadataPlane):
    """Single-process MetadataPlane. Allocates SSD extents via FileObjectStore;
    the BaM runtime reads/writes the bytes at those locations separately.
    """

    def __init__(self, object_store: FileObjectStore):
        self._store = object_store
        self._records: dict[ObjectKey, MetadataRecord] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _transition(self, rec: MetadataRecord, dst: State) -> None:
        validate_transition(rec.state, dst)
        rec.state = dst

    def lookup_meta(self, keys):
        out: dict[ObjectKey, MetadataRecord] = {}
        with self._lock:
            for k in keys:
                rec = self._records.get(k)
                if rec is not None and rec.state is State.READY:
                    rec.last_access_ts = time.monotonic()
                    out[k] = rec
                    self._hits += 1
                else:
                    self._misses += 1
        return out

    def reserve_store(self, sizes):
        out: dict[ObjectKey, SsdLocation] = {}
        with self._lock:
            for k, nbytes in sizes.items():
                existing = self._records.get(k)
                if existing is not None and existing.state is State.READY:
                    # Already published; reuse its location, caller may skip write.
                    out[k] = SsdLocation(existing.ssd_object_id, existing.offset, existing.nbytes)
                    continue
                loc = self._store.allocate(nbytes)
                rec = MetadataRecord(object_key=k, state=State.MISSING)
                self._transition(rec, State.WRITING)
                rec.ssd_object_id = loc.object_id
                rec.offset = loc.offset
                rec.nbytes = loc.nbytes
                self._records[k] = rec
                out[k] = loc
        return out

    def commit_store(self, keys):
        with self._lock:
            for k in keys:
                rec = self._records[k]
                if rec.state is State.READY:
                    continue
                self._transition(rec, State.READY)
                rec.version += 1

    def abort_store(self, keys):
        with self._lock:
            for k in keys:
                rec = self._records.get(k)
                if rec is None or rec.state is State.ABORTED:
                    continue
                self._transition(rec, State.ABORTED)

    def begin_load(self, keys):
        out: dict[ObjectKey, MetadataRecord] = {}
        with self._lock:
            for k in keys:
                rec = self._records[k]
                if rec.state is State.READY:
                    self._transition(rec, State.LOADING)
                rec.ref_count += 1
                out[k] = rec
        return out

    def release(self, keys):
        with self._lock:
            for k in keys:
                rec = self._records[k]
                rec.ref_count = max(0, rec.ref_count - 1)
                if rec.ref_count == 0 and rec.state is State.LOADING:
                    self._transition(rec, State.READY)

    def mark_failed(self, keys, reason):
        with self._lock:
            for k in keys:
                rec = self._records.get(k)
                if rec is None or rec.state is State.FAILED:
                    continue
                self._transition(rec, State.FAILED)
                rec.ref_count = 0
                rec.failure_reason = reason

    def pin(self, keys):
        with self._lock:
            for k in keys:
                self._records[k].pinned = True

    def unpin(self, keys):
        with self._lock:
            for k in keys:
                self._records[k].pinned = False

    def evict_ready(self, limit):
        evicted: list[ObjectKey] = []
        with self._lock:
            for k, rec in list(self._records.items()):
                if len(evicted) >= limit:
                    break
                if rec.state is State.READY and rec.ref_count == 0 and not rec.pinned:
                    self._transition(rec, State.EVICTED)
                    evicted.append(k)
        return evicted

    def stats(self):
        with self._lock:
            c = Counter(r.state for r in self._records.values())
            return MetadataStats(
                total=len(self._records),
                ready=c[State.READY],
                writing=c[State.WRITING],
                loading=c[State.LOADING],
                failed=c[State.FAILED],
                evicted=c[State.EVICTED],
                hits=self._hits,
                misses=self._misses,
            )
```

- [ ] **Step 3: Create `tests/test_metadata_store.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_metadata_store.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bam_kv_cache/metadata/api.py src/bam_kv_cache/metadata/store.py tests/test_metadata_store.py
git commit -m "feat(metadata): MetadataPlane API + in-process store"
```

---

## Task 7: BamConnectorCore + round-trip test

**Files:**
- Create: `src/bam_kv_cache/connector/vllm_interface.py`
- Create: `src/bam_kv_cache/connector/bam_connector.py`
- Test: `tests/test_connector_roundtrip.py`

- [ ] **Step 1: Create `src/bam_kv_cache/connector/vllm_interface.py`**

```python
from abc import ABC, abstractmethod
from typing import Any


class KVConnectorLike(ABC):
    """Thin local mirror of the vLLM KVConnectorBase_V1 hook subset we use.

    Deliberately does NOT import vllm: the vendored vllm does not import on a
    CPU-only machine, and isolating these signatures keeps the connector core
    testable. A real adapter subclassing KVConnectorBase_V1 and delegating to
    BamConnectorCore is added later in the GPU environment.
    """

    @abstractmethod
    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_num_new_matched_tokens(self, request: Any) -> int: ...

    @abstractmethod
    def start_load_kv(self, request: Any) -> None: ...

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None: ...

    @abstractmethod
    def save_kv_layer(self, layer_name: str, request: Any) -> None: ...

    @abstractmethod
    def wait_for_save(self) -> None: ...

    @abstractmethod
    def request_finished(self, request: Any) -> None: ...
```

- [ ] **Step 2: Create `src/bam_kv_cache/connector/bam_connector.py`**

```python
from dataclasses import dataclass
from typing import Any

from ..config import KVConfig
from ..metadata.api import MetadataPlane
from ..metadata.keys import ObjectKey, build_object_key
from ..runtime.base import BamRuntime
from ..runtime.descriptors import IoHandle, IoStatus, KVTensorDescriptor, SlotMapping
from ..ssd.object_store import SsdLocation
from .vllm_interface import KVConnectorLike


@dataclass
class KVChunk:
    """One token chunk of a request. `layer_tensors` maps layer_name -> the KV
    tensor for that (chunk, layer). For load, these are the destination buffers.
    """

    chunk_hash: str
    layer_tensors: dict[str, Any]  # layer_name -> torch.Tensor


@dataclass
class KVRequest:
    request_id: str
    chunks: list[KVChunk]


class BamConnectorError(RuntimeError):
    pass


class BamConnectorCore(KVConnectorLike):
    """Translates vLLM connector hooks into MetadataPlane queries and BaM I/O.

    Depends only on KVConfig + MetadataPlane + BamRuntime. Layer id is the
    registration order index of the layer name.
    """

    def __init__(self, config: KVConfig, metadata: MetadataPlane, runtime: BamRuntime):
        self._cfg = config
        self._meta = metadata
        self._rt = runtime
        self._layer_names: list[str] = []
        self._save_handles: list[tuple[IoHandle, ObjectKey]] = []
        self._load_handles: dict[str, list[tuple[IoHandle, ObjectKey]]] = {}
        self._held_refs: list[ObjectKey] = []

    # ---- registration ----
    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._layer_names = list(kv_caches.keys())

    def _layer_id(self, layer_name: str) -> int:
        return self._layer_names.index(layer_name)

    def _key(self, chunk_hash: str, layer_name: str) -> ObjectKey:
        return build_object_key(
            model_id=self._cfg.model_id,
            tokenizer_id=self._cfg.tokenizer_id,
            token_chunk_hash=chunk_hash,
            layer_id=self._layer_id(layer_name),
            tp_rank=self._cfg.tp_rank,
            kv_dtype=self._cfg.kv_dtype,
            kv_layout=self._cfg.kv_layout,
            chunk_size=self._cfg.chunk_size,
        )

    # ---- store path ----
    def save_kv_layer(self, layer_name: str, request: KVRequest) -> None:
        for chunk in request.chunks:
            tensor = chunk.layer_tensors[layer_name]
            key = self._key(chunk.chunk_hash, layer_name)
            nbytes = tensor.element_size() * tensor.nelement()
            loc = self._meta.reserve_store({key: nbytes})[key]
            kv = KVTensorDescriptor(tensor=tensor, layer_id=self._layer_id(layer_name), nbytes=nbytes)
            handle = self._rt.submit_store(kv, SlotMapping(0, nbytes), loc)
            self._save_handles.append((handle, key))

    def wait_for_save(self) -> None:
        for handle, key in self._save_handles:
            if self._rt.wait(handle) is IoStatus.DONE:
                self._meta.commit_store([key])
            else:
                self._meta.abort_store([key])
        self._save_handles.clear()

    # ---- load path ----
    def get_num_new_matched_tokens(self, request: KVRequest) -> int:
        matched_chunks = 0
        for chunk in request.chunks:
            keys = [self._key(chunk.chunk_hash, ln) for ln in self._layer_names]
            found = self._meta.lookup_meta(keys)
            if keys and all(k in found for k in keys):
                matched_chunks += 1
        return matched_chunks * self._cfg.chunk_size

    def start_load_kv(self, request: KVRequest) -> None:
        for chunk in request.chunks:
            keys = [self._key(chunk.chunk_hash, ln) for ln in self._layer_names]
            found = self._meta.lookup_meta(keys)
            if not (keys and all(k in found for k in keys)):
                continue  # not a full hit; no partial load in v1
            records = self._meta.begin_load(keys)
            for ln in self._layer_names:
                key = self._key(chunk.chunk_hash, ln)
                rec = records[key]
                loc = SsdLocation(rec.ssd_object_id, rec.offset, rec.nbytes)
                kv = KVTensorDescriptor(
                    tensor=chunk.layer_tensors[ln], layer_id=self._layer_id(ln), nbytes=rec.nbytes
                )
                handle = self._rt.submit_load(kv, SlotMapping(0, rec.nbytes), loc)
                self._load_handles.setdefault(ln, []).append((handle, key))
                self._held_refs.append(key)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for handle, key in self._load_handles.get(layer_name, []):
            if self._rt.wait(handle) is IoStatus.DONE:
                self._meta.release([key])
                self._held_refs.remove(key)
            else:
                # fail-fast (design §11): mark failed (clears ref) and raise.
                self._meta.mark_failed([key], "bam load failed")
                self._held_refs.remove(key)
                self._load_handles[layer_name] = []
                raise BamConnectorError(f"load failed for {key}")
        self._load_handles[layer_name] = []

    def request_finished(self, request: KVRequest) -> None:
        for key in list(self._held_refs):
            self._meta.release([key])
        self._held_refs.clear()
        self._load_handles.clear()
        self._save_handles.clear()
```

- [ ] **Step 3: Create `tests/test_connector_roundtrip.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_connector_roundtrip.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bam_kv_cache/connector tests/test_connector_roundtrip.py
git commit -m "feat(connector): BamConnectorCore with mock-runtime round-trip"
```

---

## Task 8: Full-suite green + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -v`
Expected: all tests pass (smoke 2, keys 6, state_machine 6, object_store 4, mock_runtime 3, metadata_store 9, connector 4).

- [ ] **Step 2: Update `README.md`**

```markdown
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
```

- [ ] **Step 3: Run suite once more to confirm nothing broke**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README for phases 1-2"
```

---

## Self-review notes (already applied)

- **Spec coverage:** keys (§6) → Task 2; record/state machine (§4.1, §7.3) → Task 3; SSD store (§3) → Task 4; `BamRuntime`+mock (§4.2) → Task 5; `MetadataPlane`+in-process (§4.1) → Task 6; `BamConnectorCore` store/load (§5) → Task 7; completion criteria (§8) → Tasks 6–7 tests. No `vllm` import anywhere (Task 7 uses local `KVConnectorLike`).
- **Type consistency:** `ObjectKey`, `SsdLocation`, `KVTensorDescriptor`, `SlotMapping`, `IoHandle`/`IoStatus`, `MetadataRecord`/`State`, `KVConfig`, `KVChunk`/`KVRequest` are each defined once and imported consistently. `reserve_store(sizes: dict)`, `begin_load -> dict[key, record]`, `mark_failed(keys, reason)` signatures match between `MetadataPlane` (Task 6) and `BamConnectorCore` (Task 7).
- **Known v1 simplifications (documented):** `reserved`+`writing` collapsed into `WRITING`; whole-tensor-per-(chunk,layer) instead of real paged-slot slicing; no partial-hit load; synchronous mock I/O.
