#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bam_runtime.h"

namespace py = pybind11;
using bamkv::BamDevice;
using bamkv::BamDeviceParams;

PYBIND11_MODULE(_native, m) {
    m.doc() = "BaM-backed KV cache data plane (native).";

    py::class_<BamDevice>(m, "BamDevice")
        .def(py::init([](std::vector<std::string> nvme_paths, int nvm_namespace, int cuda_device,
                         int queue_depth, int num_queues, int64_t page_size,
                         int64_t page_cache_pages, int64_t lba_block_size) {
                 BamDeviceParams p;
                 p.nvme_paths = std::move(nvme_paths);
                 p.nvm_namespace = nvm_namespace;
                 p.cuda_device = cuda_device;
                 p.queue_depth = queue_depth;
                 p.num_queues = num_queues;
                 p.page_size = page_size;
                 p.page_cache_pages = page_cache_pages;
                 p.lba_block_size = lba_block_size;
                 return new BamDevice(p);
             }),
             py::arg("nvme_paths"), py::arg("nvm_namespace"), py::arg("cuda_device"),
             py::arg("queue_depth"), py::arg("num_queues"), py::arg("page_size"),
             py::arg("page_cache_pages"), py::arg("lba_block_size"))
        .def("submit_store", &BamDevice::submit_store,
             py::arg("dptr"), py::arg("nbytes"), py::arg("dev_offset"), py::arg("stream"))
        .def("submit_load", &BamDevice::submit_load,
             py::arg("dptr"), py::arg("nbytes"), py::arg("dev_offset"), py::arg("stream"))
        .def("poll", &BamDevice::poll, py::arg("handle"))
        .def("wait", &BamDevice::wait, py::arg("handle"));
}
