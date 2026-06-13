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
