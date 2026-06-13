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
            if self._next_offset > 0:  # nothing to extend for a zero-byte alloc
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
