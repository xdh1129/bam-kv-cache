# Phase 3: building and running BaM runtime on the GPU box

## Prereqs

One-time machine setup:

- IOMMU disabled.
- Above-4G Decoding and Resizable BAR enabled.
- `nvidia-smi -q -d MEMORY | grep -iA3 BAR1` shows GB-scale BAR1 memory.
- BaM is built, with `$BAM_HOME/build/lib/libnvm.so` and `$BAM_HOME/include`.
- BaM kernel module is loaded and `/dev/libnvm0` exists.
- `/dev/nvme0n1` is a wipeable spare namespace bound to BaM.

## Build the native extension

```bash
export BAM_HOME=/path/to/bam
export BAM_KV_BUILD_NATIVE=1
export BAM_KV_CUDA_ARCH=89
pip install -e ".[test,native]"
python -c "import bam_kv_cache._native; print('native OK')"
```

`BAM_KV_CUDA_ARCH=89` targets RTX 6000 Ada. Change it if the GPU box uses a
different compute capability.

## Run the hardware round-trip

```bash
export BAM_KV_RUN_HW=1
export BAM_KV_NVME=/dev/libnvm0
python -m pytest tests/test_native_roundtrip.py -v
```

## Local/off-box behavior

The pure-Python suite runs with or without the native build:

```bash
python -m pytest -q
```

`tests/test_native_roundtrip.py` is skipped unless `bam_kv_cache._native` is
importable, CUDA is available, and `BAM_KV_RUN_HW=1` is set.
