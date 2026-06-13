import enum
import time
from dataclasses import dataclass, field

from .keys import ObjectKey


class State(enum.Enum):
    MISSING = "missing"
    WRITING = "writing"    # reserved + being written (collapsed, see plan note)
    READY = "ready"
    LOADING = "loading"
    ABORTED = "aborted"
    FAILED = "failed"
    EVICTED = "evicted"


@dataclass
class MetadataRecord:
    object_key: ObjectKey
    state: State
    ssd_object_id: int | None = None
    offset: int | None = None
    nbytes: int | None = None
    version: int = 0
    ref_count: int = 0
    pinned: bool = False
    last_access_ts: float = field(default_factory=time.monotonic)
    failure_reason: str | None = None
