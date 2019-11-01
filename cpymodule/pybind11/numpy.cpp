#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

py::array_t<double> make_array(const py::ssize_t size) {
    // No pointer is passed, so NumPy will allocate the buffer
    return py::array_t<double>(size);
}

PYBIND11_MODULE(nump, m) {
    m.doc() = "pybind11 numpy plugin";
    m.def("make_array", &make_array, "Functiony",
          py::return_value_policy::move); // Return policy can be left default, i.e. return_value_policy::automatic
}
