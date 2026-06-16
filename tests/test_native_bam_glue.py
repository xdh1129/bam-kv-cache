from bam_kv_cache.runtime.descriptors import IoStatus, KVTensorDescriptor, SlotMapping
from bam_kv_cache.runtime.native_bam import NativeBamRuntime
from bam_kv_cache.ssd.object_store import SsdLocation


class _FakeTensor:
    def __init__(self, ptr, nbytes):
        self._ptr = ptr
        self._n = nbytes

    def data_ptr(self):
        return self._ptr


class _FakeDevice:
    """Records calls and returns scripted statuses."""

    def __init__(self):
        self.calls = []
        self._next = 1
        self.status_for = {}

    def submit_store(self, dptr, nbytes, dev_offset, stream):
        self.calls.append(("store", dptr, nbytes, dev_offset, stream))
        h = self._next
        self._next += 1
        return h

    def submit_load(self, dptr, nbytes, dev_offset, stream):
        self.calls.append(("load", dptr, nbytes, dev_offset, stream))
        h = self._next
        self._next += 1
        return h

    def poll(self, h):
        return self.status_for.get(h, 0)

    def wait(self, h):
        return self.status_for.get(h, 1)


def _kv(ptr=0xCAFE, nbytes=4096):
    return KVTensorDescriptor(tensor=_FakeTensor(ptr, nbytes), layer_id=0, nbytes=nbytes)


def test_submit_store_marshals_fields():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=0, offset=8192, nbytes=4096)
    h = rt.submit_store(_kv(0xABCD, 4096), SlotMapping(0, 4096), loc, stream=7)
    assert dev.calls[0] == ("store", 0xABCD, 4096, 8192, 7)
    assert h.handle_id == 1


def test_submit_load_marshals_fields():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=1, offset=0, nbytes=4096)
    rt.submit_load(_kv(0x1000, 4096), SlotMapping(0, 4096), loc, stream=3)
    assert dev.calls[0] == ("load", 0x1000, 4096, 0, 3)


def test_poll_and_wait_status_mapping():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=0, offset=0, nbytes=4096)
    h = rt.submit_store(_kv(), SlotMapping(0, 4096), loc, stream=0)
    dev.status_for[h.handle_id] = 0
    assert rt.poll(h) is IoStatus.PENDING
    dev.status_for[h.handle_id] = 1
    assert rt.wait(h) is IoStatus.DONE


def test_wait_failure_sets_error():
    dev = _FakeDevice()
    rt = NativeBamRuntime(dev)
    loc = SsdLocation(object_id=0, offset=0, nbytes=4096)
    h = rt.submit_load(_kv(), SlotMapping(0, 4096), loc, stream=0)
    dev.status_for[h.handle_id] = 2
    assert rt.wait(h) is IoStatus.FAILED
    assert h.error is not None
