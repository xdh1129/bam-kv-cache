from abc import ABC, abstractmethod
from typing import Any


class KVConnectorLike(ABC):
    """Thin local mirror of the vLLM KVConnectorBase_V1 hook subset we use.

    Deliberately does NOT import vllm: the vendored vllm does not import on a
    CPU-only machine, and isolating these signatures keeps the connector core
    testable. A real adapter subclassing KVConnectorBase_V1 and delegating to
    BamConnectorCore is added later in the GPU environment.
    """

    @abstractmethod
    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_num_new_matched_tokens(self, request: Any) -> int: ...

    @abstractmethod
    def start_load_kv(self, request: Any) -> None: ...

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None: ...

    @abstractmethod
    def save_kv_layer(self, layer_name: str, request: Any) -> None: ...

    @abstractmethod
    def wait_for_save(self) -> None: ...

    @abstractmethod
    def request_finished(self, request: Any) -> None: ...
