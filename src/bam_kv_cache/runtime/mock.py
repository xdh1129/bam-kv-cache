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
