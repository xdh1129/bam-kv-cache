import os
import importlib.util

import pytest

try:
    import torch
except ImportError:
    torch = None

from bam_kv_cache.config import KVConfig
from bam_kv_cache.config_bam import BamDeviceConfig
from bam_kv_cache.connector.bam_connector import BamConnectorCore, KVChunk, KVRequest
from bam_kv_cache.metadata.store import InProcessMetadataStore
from bam_kv_cache.runtime.descriptors import IoStatus, KVTensorDescriptor, SlotMapping
from bam_kv_cache.runtime.lba_alloc import RawNamespaceAllocator
from bam_kv_cache.runtime.native_bam import NativeBamRuntime

_HAS_NATIVE = importlib.util.find_spec("bam_kv_cache._native") is not None
_HAS_CUDA = bool(torch is not None and torch.cuda.is_available())
_RUN_HW = os.environ.get("BAM_KV_RUN_HW") == "1"

pytestmark = pytest.mark.skipif(
    not (_HAS_NATIVE and _HAS_CUDA and _RUN_HW),
    reason="requires bam_kv_cache._native, CUDA, and BAM_KV_RUN_HW=1",
)


def _dev_cfg():
    return BamDeviceConfig(
        nvme_paths=tuple(os.environ.get("BAM_KV_NVME", "/dev/libnvm0").split(",")),
        nvm_namespace=1,
        cuda_device=0,
        queue_depth=1024,
        num_queues=128,
        page_size=4096,
        page_cache_pages=65536,
        lba_block_size=512,
        namespace_capacity_bytes=2048408248320,
    )


def test_runtime_roundtrip_single_extent():
    dcfg = _dev_cfg()
    rt = NativeBamRuntime.from_config(dcfg)
    alloc = RawNamespaceAllocator(dcfg.namespace_capacity_bytes, dcfg.lba_block_size)

    src = torch.arange(1024, dtype=torch.float32, device="cuda")
    nbytes = src.element_size() * src.nelement()
    loc = alloc.allocate(nbytes)

    hs = rt.submit_store(KVTensorDescriptor(src, 0, nbytes), SlotMapping(0, nbytes), loc)
    assert rt.wait(hs) is IoStatus.DONE

    dst = torch.zeros_like(src)
    hl = rt.submit_load(KVTensorDescriptor(dst, 0, nbytes), SlotMapping(0, nbytes), loc)
    assert rt.wait(hl) is IoStatus.DONE
    assert torch.equal(dst, src)


def test_connector_roundtrip_on_bam():
    dcfg = _dev_cfg()
    rt = NativeBamRuntime.from_config(dcfg)
    meta = InProcessMetadataStore(
        RawNamespaceAllocator(dcfg.namespace_capacity_bytes, dcfg.lba_block_size)
    )
    kcfg = KVConfig(
        model_id="m",
        tokenizer_id="t",
        kv_dtype="float32",
        kv_layout="NHD",
        chunk_size=16,
        tp_rank=0,
        ssd_root="",
    )
    conn = BamConnectorCore(kcfg, meta, rt)
    layers = ["layer.0", "layer.1"]
    conn.register_kv_caches({ln: None for ln in layers})

    def mk(fill):
        return KVRequest(
            request_id="r",
            chunks=[
                KVChunk(
                    chunk_hash="c0",
                    layer_tensors={
                        ln: (torch.arange(1024, dtype=torch.float32, device="cuda") + i + fill)
                        for i, ln in enumerate(layers)
                    },
                )
            ],
        )

    src = mk(0.0)
    for ln in layers:
        conn.save_kv_layer(ln, src)
    conn.wait_for_save()

    dst = KVRequest(
        request_id="r2",
        chunks=[
            KVChunk(
                chunk_hash="c0",
                layer_tensors={ln: torch.zeros(1024, device="cuda") for ln in layers},
            )
        ],
    )
    assert conn.get_num_new_matched_tokens(dst) == 16
    conn.start_load_kv(dst)
    for ln in layers:
        conn.wait_for_layer_load(ln)
    for ln in layers:
        assert torch.equal(dst.chunks[0].layer_tensors[ln], src.chunks[0].layer_tensors[ln])
    conn.request_finished(dst)
