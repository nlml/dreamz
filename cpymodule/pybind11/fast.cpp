#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

// C++ code
void calc_sum_cost(float* ptr, int N, int M, float* ptr_cost) {
  for(int32_t i = 1; i < N; i++) {
    for(int32_t j = 1; j < M; j++) {
      float upc = ptr[(i-1) * M + j];
      float leftc = ptr[i * M + j - 1];
      float diagc = ptr[(i-1) * M + j - 1];
      float transition_cost = std::min(upc, std::min(leftc, diagc));
      if (transition_cost == diagc) {
        transition_cost += 2 * ptr_cost[i*M + j];
      } else {
        transition_cost += ptr_cost[i*M + j];
      }
      std::cout << transition_cost << std::endl;
      ptr[i * M + j] = transition_cost;
    }
  }
}

// Interface

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::object wrapper(py::array_t<float> array,
                  py::array_t<float> arrayb) {
  // check input dimensions
  if ( array.ndim()     != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");

  auto buf = array.request();
  auto buf2 = arrayb.request();
  if (buf.size != buf2.size) throw std::runtime_error("sizes do not match!");

  int N = array.shape()[0], M = array.shape()[1];

  float* ptr = (float*) buf.ptr;
  float* ptr_cost = (float*) buf2.ptr;
  // call pure C++ function
  calc_sum_cost(ptr, N, M, ptr_cost);
  return py::cast<py::none>(Py_None);
}

PYBIND11_MODULE(fast,m) {
  m.doc() = "pybind11 plugin";
  m.def("calc_sum_cost", &wrapper, "Calculate the length of an array of vectors");
}
