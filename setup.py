import os

from setuptools import setup

ext_modules = []
cmdclass = {}

if os.environ.get("BAM_KV_BUILD_NATIVE") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    bam_home = os.environ.get("BAM_HOME")
    if not bam_home:
        raise RuntimeError("BAM_KV_BUILD_NATIVE=1 requires BAM_HOME to point at the built BaM repo")
    arch = os.environ.get("BAM_KV_CUDA_ARCH", "89")  # RTX 6000 Ada = sm_89

    nvcc_args = ["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    # Pin nvcc's host compiler. BaM's bundled freestanding headers reject newer
    # g++ frontends (e.g. g++-13), so libnvm must be built with an older g++
    # (g++-12 works); nvcc has to use that same host compiler or it falls back to
    # the system default and hits the freestanding error. Set BAM_KV_HOST_CXX (or
    # CXX) to that compiler, e.g. BAM_KV_HOST_CXX=g++-12.
    host_cxx = os.environ.get("BAM_KV_HOST_CXX") or os.environ.get("CXX")
    if host_cxx:
        nvcc_args += ["-ccbin", host_cxx]

    ext_modules = [
        CUDAExtension(
            name="bam_kv_cache._native",
            sources=["native/bindings.cpp", "native/bam_runtime.cu"],
            include_dirs=[os.path.join(bam_home, "include")],
            library_dirs=[os.path.join(bam_home, "build", "lib")],
            libraries=["nvm"],
            runtime_library_dirs=[os.path.join(bam_home, "build", "lib")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_args,
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
