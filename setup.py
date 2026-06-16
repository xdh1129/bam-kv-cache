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
                "nvcc": ["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
