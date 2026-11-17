// pybind11 bindings: numpy (T, N) float32 <-> Panel. Built only when pybind11 is found.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "sutra/ops.hpp"

namespace py = pybind11;
using sutra::Panel;

static Panel from_numpy(py::array_t<float, py::array::f_style | py::array::forcecast> a) {
    Panel p(a.shape(0), a.shape(1));
    std::copy(a.data(), a.data() + a.size(), p.data.begin());  // f_style == column-major
    return p;
}

static py::array_t<float> to_numpy(const Panel& p) {
    py::array_t<float, py::array::f_style> a({p.T, p.N});
    std::copy(p.data.begin(), p.data.end(), a.mutable_data());
    return a;
}

PYBIND11_MODULE(_core, m) {
    m.def("ts_mean", [](py::array_t<float, py::array::f_style | py::array::forcecast> x, int d) {
        return to_numpy(sutra::ts_mean(from_numpy(x), d));
    });
    m.def("ts_delta", [](py::array_t<float, py::array::f_style | py::array::forcecast> x, int d) {
        return to_numpy(sutra::ts_delta(from_numpy(x), d));
    });
    m.def("rank", [](py::array_t<float, py::array::f_style | py::array::forcecast> x) {
        return to_numpy(sutra::rank(from_numpy(x)));
    });
}
