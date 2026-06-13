import enum
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVTensorDescriptor:
    """Describes a KV tensor region to move.

    In v1 `tensor` is a CPU torch.Tensor. The real runtime will instead carry a
    GPU pointer / paged-buffer descriptor; consumers only rely on the fields
    below, not on `tensor` being host memory.
    """

    tensor: Any        # torch.Tensor (CPU in mock)
    layer_id: int
    nbytes: int


@dataclass(frozen=True)
class SlotMapping:
    """Paged-KV slot mapping. v1 uses a single contiguous byte range."""

    start: int
    length: int


class IoStatus(enum.Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


@dataclass
class IoHandle:
    handle_id: int
    status: IoStatus = IoStatus.PENDING
    error: str | None = None
