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
