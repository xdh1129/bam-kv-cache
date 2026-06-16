# Phase 3: building and running the BaM runtime on the GPU box

This is the proven recipe from bringing the native extension up on
`aiserver-nv-6000ada-x8-122` (Ubuntu 24.04, kernel 6.8, CUDA 13, RTX 6000 Ada
sm_89). Adjust paths/versions for your box.

## 0. Toolchain reality check

BaM's bundled `freestanding` headers reject newer g++ frontends, so the whole
stack must be built with an **older host compiler**. On this box the system g++
is 13 (fails); **g++-12 works**:

```bash
sudo apt install -y gcc-12 g++-12 cmake ninja-build
ls -d /usr/local/cuda*            # CUDA toolkit (13.x here); add bin/ to PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
nvcc --version
```

## 1. Build BaM's `libnvm.so` (no root)

```bash
cd $BAM_HOME                      # the BaM repo
git submodule update --init --recursive    # pulls include/freestanding (provides <simt/atomic>)
mkdir -p build && cd build
rm -f CMakeCache.txt              # if re-running with a different compiler
CC=gcc-12 CXX=g++-12 cmake ..
make libnvm -j
ls lib/libnvm.so                  # expected artifact
```

## 2. Build the native extension (no root)

Requires a CUDA-enabled torch whose CUDA major matches `nvcc`
(here torch 2.12+cu130 vs nvcc 13 — major 13 matches). Use a venv; Ubuntu 24.04
blocks system `pip install`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade "setuptools<82" wheel pybind11 ninja numpy torch

export BAM_HOME=/home/poc/bam-connector/bam
export BAM_KV_BUILD_NATIVE=1
export BAM_KV_CUDA_ARCH=89
export BAM_KV_HOST_CXX=g++-12     # nvcc host compiler; must match step 1
export CC=gcc-12 CXX=g++-12
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

# --no-build-isolation so setup.py can see the venv's torch
pip install -e ".[test,native]" --no-build-isolation

python -c "import torch, bam_kv_cache._native; print('native OK')"
```

Notes:
- `import torch` must come before `import bam_kv_cache._native` (the ext links
  libtorch/libc10). `runtime/native_bam.py` imports torch at module top for this.
- `BAM_KV_HOST_CXX` adds `-ccbin g++-12` to nvcc; without it nvcc falls back to
  the system g++ and re-hits the freestanding error.
- `setup.py` mirrors BaM's three include dirs: `include`,
  `include/freestanding/include`, `build/include`.

## 3. Load the BaM kernel module (root) — the kernel-6.x risk

BaM's README is tested on kernel 5.8 and warns 6.x may not work. In practice,
on kernel **6.8** the module compiles after **two one-line API ports** (the
nvidia P2P symbols themselves are still exported by driver 580, so MODPOST is
clean):

- `module/pci.c` — `class_create()` lost its `owner` arg in 6.4:
  `class_create(THIS_MODULE, DRIVER_NAME)` → `class_create(DRIVER_NAME)`.
- `module/map.c` — `get_user_pages()` lost its trailing `vmas` arg in 6.5:
  `get_user_pages(..., FOLL_WRITE, pages, NULL)` → `get_user_pages(..., FOLL_WRITE, pages)`.

The nvidia driver's `Module.symvers` must exist first (cmake only enables the
`_CUDA`/P2P module build when it does), so build the driver symbols, then
re-run cmake so it picks them up:

```bash
# Build nvidia driver kernel symbols (generates Module.symvers; compile-only,
# does NOT load or replace the running driver). version = your driver.
cd /usr/src/nvidia-<DRIVER_VERSION>/ && sudo make
cd $BAM_HOME/build && CC=gcc-12 CXX=g++-12 cmake ..   # now detects Module.symvers

# Apply the two 6.x ports, then build the module.
# NOTE: build the module with the kernel's own compiler (gcc-13 here), NOT g++-12,
# so unset CC/CXX first.
cd $BAM_HOME/build/module && unset CC CXX && make     # -> libnvm.ko

# Find the spare NVMe's PCI ID (use a WIPEABLE namespace)
dmesg | grep nvme0

# Unbind it from the kernel nvme driver (root)
echo -n "<PCI_ID>" | sudo tee /sys/bus/pci/devices/<PCI_ID>/driver/unbind

# Load BaM module -> creates /dev/libnvm*
cd $BAM_HOME/build/module && sudo make load
ls -l /dev/libnvm*
```

Also confirm IOMMU is off: `cat /proc/cmdline | grep -i iommu` should find
nothing (else disable via grub + reboot).

## 4. Run the hardware round-trip

> WARNING: this writes raw LBAs to the bound NVMe namespace from offset 0.
> Use a spare, wipeable namespace.

```bash
export BAM_KV_RUN_HW=1
export BAM_KV_NVME=/dev/libnvm0
python -m pytest tests/test_native_roundtrip.py -v
```

## Local / off-box behavior

The pure-Python suite runs with or without the native build:

```bash
python -m pytest -q          # 47 passed, 2 skipped
```

`tests/test_native_roundtrip.py` is skipped unless `bam_kv_cache._native` is
importable, CUDA is available, and `BAM_KV_RUN_HW=1` is set.
