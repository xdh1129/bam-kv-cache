from typing import Any

# Importing torch first loads libc10/libtorch into the process so the native
# extension (which links against them) can resolve its symbols on dlopen.
import torch  # noqa: F401

from .base import BamRuntime
from .descriptors import IoHandle, IoStatus, KVTensorDescriptor, SlotMapping

_STATUS = {0: IoStatus.PENDING, 1: IoStatus.DONE, 2: IoStatus.FAILED}


def _stream_ptr(stream: Any) -> int:
    """Return a cudaStream_t as int. None maps to torch's current CUDA stream."""
    if stream is None:
        import torch

        return torch.cuda.current_stream().cuda_stream
    if hasattr(stream, "cuda_stream"):
        return stream.cuda_stream
    return int(stream)


class NativeBamRuntime(BamRuntime):
    """BamRuntime backed by the native BaM device extension."""

    def __init__(self, device: Any):
        self._dev = device

    @classmethod
    def from_config(cls, cfg) -> "NativeBamRuntime":
        from bam_kv_cache import _native

        dev = _native.BamDevice(
            nvme_paths=list(cfg.nvme_paths),
            nvm_namespace=cfg.nvm_namespace,
            cuda_device=cfg.cuda_device,
            queue_depth=cfg.queue_depth,
            num_queues=cfg.num_queues,
            page_size=cfg.page_size,
            page_cache_pages=cfg.page_cache_pages,
            lba_block_size=cfg.lba_block_size,
        )
        return cls(dev)

    def submit_store(
        self,
        kv: KVTensorDescriptor,
        slot_mapping: SlotMapping,
        location,
        stream=None,
    ) -> IoHandle:
        hid = self._dev.submit_store(
            kv.tensor.data_ptr(), kv.nbytes, location.offset, _stream_ptr(stream)
        )
        return IoHandle(handle_id=hid)

    def submit_load(
        self,
        kv: KVTensorDescriptor,
        slot_mapping: SlotMapping,
        location,
        stream=None,
    ) -> IoHandle:
        hid = self._dev.submit_load(
            kv.tensor.data_ptr(), kv.nbytes, location.offset, _stream_ptr(stream)
        )
        return IoHandle(handle_id=hid)

    def poll(self, handle: IoHandle) -> IoStatus:
        st = _STATUS.get(self._dev.poll(handle.handle_id), IoStatus.FAILED)
        handle.status = st
        return st

    def wait(self, handle: IoHandle) -> IoStatus:
        st = _STATUS.get(self._dev.wait(handle.handle_id), IoStatus.FAILED)
        handle.status = st
        if st is IoStatus.FAILED:
            handle.error = "bam device reported failure"
        return st
