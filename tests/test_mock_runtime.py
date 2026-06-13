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
