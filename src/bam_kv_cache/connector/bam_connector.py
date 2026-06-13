from dataclasses import dataclass
from typing import Any

from ..config import KVConfig
from ..metadata.api import MetadataPlane
from ..metadata.keys import ObjectKey, build_object_key
from ..runtime.base import BamRuntime
from ..runtime.descriptors import IoHandle, IoStatus, KVTensorDescriptor, SlotMapping
from ..ssd.object_store import SsdLocation
from .vllm_interface import KVConnectorLike


@dataclass
class KVChunk:
    """One token chunk of a request. `layer_tensors` maps layer_name -> the KV
    tensor for that (chunk, layer). For load, these are the destination buffers.
    """

    chunk_hash: str
    layer_tensors: dict[str, Any]  # layer_name -> torch.Tensor


@dataclass
class KVRequest:
    request_id: str
    chunks: list[KVChunk]


class BamConnectorError(RuntimeError):
    pass


class BamConnectorCore(KVConnectorLike):
    """Translates vLLM connector hooks into MetadataPlane queries and BaM I/O.

    Depends only on KVConfig + MetadataPlane + BamRuntime. Layer id is the
    registration order index of the layer name.
    """

    def __init__(self, config: KVConfig, metadata: MetadataPlane, runtime: BamRuntime):
        self._cfg = config
        self._meta = metadata
        self._rt = runtime
        self._layer_names: list[str] = []
        self._save_handles: list[tuple[IoHandle, ObjectKey]] = []
        self._load_handles: dict[str, list[tuple[IoHandle, ObjectKey]]] = {}
        self._held_refs: list[ObjectKey] = []

    # ---- registration ----
    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._layer_names = list(kv_caches.keys())

    def _layer_id(self, layer_name: str) -> int:
        return self._layer_names.index(layer_name)

    def _key(self, chunk_hash: str, layer_name: str) -> ObjectKey:
        return build_object_key(
            model_id=self._cfg.model_id,
            tokenizer_id=self._cfg.tokenizer_id,
            token_chunk_hash=chunk_hash,
            layer_id=self._layer_id(layer_name),
            tp_rank=self._cfg.tp_rank,
            kv_dtype=self._cfg.kv_dtype,
            kv_layout=self._cfg.kv_layout,
            chunk_size=self._cfg.chunk_size,
        )

    # ---- store path ----
    def save_kv_layer(self, layer_name: str, request: KVRequest) -> None:
        for chunk in request.chunks:
            tensor = chunk.layer_tensors[layer_name]
            key = self._key(chunk.chunk_hash, layer_name)
            nbytes = tensor.element_size() * tensor.nelement()
            loc = self._meta.reserve_store({key: nbytes})[key]
            kv = KVTensorDescriptor(tensor=tensor, layer_id=self._layer_id(layer_name), nbytes=nbytes)
            handle = self._rt.submit_store(kv, SlotMapping(0, nbytes), loc)
            self._save_handles.append((handle, key))

    def wait_for_save(self) -> None:
        for handle, key in self._save_handles:
            if self._rt.wait(handle) is IoStatus.DONE:
                self._meta.commit_store([key])
            else:
                self._meta.abort_store([key])
        self._save_handles.clear()

    # ---- load path ----
    def get_num_new_matched_tokens(self, request: KVRequest) -> int:
        matched_chunks = 0
        for chunk in request.chunks:
            keys = [self._key(chunk.chunk_hash, ln) for ln in self._layer_names]
            found = self._meta.lookup_meta(keys)
            if keys and all(k in found for k in keys):
                matched_chunks += 1
        return matched_chunks * self._cfg.chunk_size

    def start_load_kv(self, request: KVRequest) -> None:
        for chunk in request.chunks:
            keys = [self._key(chunk.chunk_hash, ln) for ln in self._layer_names]
            found = self._meta.lookup_meta(keys)
            if not (keys and all(k in found for k in keys)):
                continue  # not a full hit; no partial load in v1
            records = self._meta.begin_load(keys)
            for ln in self._layer_names:
                key = self._key(chunk.chunk_hash, ln)
                rec = records[key]
                loc = SsdLocation(rec.ssd_object_id, rec.offset, rec.nbytes)
                kv = KVTensorDescriptor(
                    tensor=chunk.layer_tensors[ln], layer_id=self._layer_id(ln), nbytes=rec.nbytes
                )
                handle = self._rt.submit_load(kv, SlotMapping(0, rec.nbytes), loc)
                self._load_handles.setdefault(ln, []).append((handle, key))
                self._held_refs.append(key)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for handle, key in self._load_handles.get(layer_name, []):
            if self._rt.wait(handle) is IoStatus.DONE:
                self._meta.release([key])
                self._held_refs.remove(key)
            else:
                # fail-fast (design §11): mark failed (clears ref) and raise.
                self._meta.mark_failed([key], "bam load failed")
                self._held_refs.remove(key)
                self._load_handles[layer_name] = []
                raise BamConnectorError(f"load failed for {key}")
        self._load_handles[layer_name] = []

    def request_finished(self, request: KVRequest) -> None:
        for key in list(self._held_refs):
            self._meta.release([key])
        self._held_refs.clear()
        self._load_handles.clear()
        self._save_handles.clear()
