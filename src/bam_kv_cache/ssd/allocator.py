from typing import Protocol, runtime_checkable

from .object_store import SsdLocation


@runtime_checkable
class SsdAllocator(Protocol):
    """Hands out byte extents on the SSD address space.

    FileObjectStore (mock/CPU) and RawNamespaceAllocator (BaM, raw namespace)
    both satisfy this. The metadata plane depends only on `allocate`.
    """

    def allocate(self, nbytes: int) -> SsdLocation: ...
