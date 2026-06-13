import threading

from ..ssd.object_store import SsdLocation


class RawNamespaceAllocator:
    """Bump-allocates LBA-aligned byte extents over a raw NVMe namespace.

    Satisfies the SsdAllocator protocol. `offset` is always a multiple of
    `lba_block_size`, so the native runtime can compute `lba = offset //
    lba_block_size` exactly. No file/backing store — the native BaM device does
    the actual I/O at these offsets.
    """

    def __init__(self, capacity_bytes: int, lba_block_size: int, alignment: int | None = None):
        if lba_block_size <= 0:
            raise ValueError("lba_block_size must be positive")
        self._capacity = capacity_bytes
        self._lba_block_size = lba_block_size
        # Align at least to an LBA; default to the LBA size.
        self._alignment = alignment if alignment is not None else lba_block_size
        if self._alignment % lba_block_size != 0:
            raise ValueError("alignment must be a multiple of lba_block_size")
        self._lock = threading.Lock()
        self._next_offset = 0
        self._next_object_id = 0

    def _align(self, n: int) -> int:
        a = self._alignment
        return ((n + a - 1) // a) * a

    def allocate(self, nbytes: int) -> SsdLocation:
        size = self._align(nbytes)
        with self._lock:
            offset = self._next_offset
            if offset + size > self._capacity:
                raise RuntimeError(
                    f"namespace out of space: need {size} at {offset}, capacity {self._capacity}"
                )
            object_id = self._next_object_id
            self._next_object_id += 1
            self._next_offset += size
        return SsdLocation(object_id=object_id, offset=offset, nbytes=nbytes)
