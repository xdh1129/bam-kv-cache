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
