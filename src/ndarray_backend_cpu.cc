#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};

// 通用模板函数，用于逐元素操作
template <typename Op>
void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

// 通用模板函数，用于标量操作
template <typename Op>
void ScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
    size_t cnt = 0;
    std::vector<int32_t> indices(shape.size(), 0);

    while (true) {
        size_t in_pos = offset;
        for (size_t i = 0; i < shape.size(); i++) {
            in_pos += strides[i] * indices[i];
        }
        out->ptr[cnt++] = a.ptr[in_pos];

        for (int i = shape.size() - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < shape[i]) {
                break;
            }
            indices[i] = 0;
            if (i == 0) return;
        }
    }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
    size_t cnt = 0;
    std::vector<int32_t> indices(shape.size(), 0);

    while (true) {
        size_t out_pos = offset;
        for (size_t i = 0; i < shape.size(); i++) {
            out_pos += strides[i] * indices[i];
        }
        out->ptr[out_pos] = a.ptr[cnt++];

        for (int i = shape.size() - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < shape[i]) {
                break;
            }
            indices[i] = 0;
            if (i == 0) return;
        }
    }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
    std::vector<int32_t> indices(shape.size(), 0);

    for (size_t cnt = 0; cnt < size; ++cnt) {
        size_t out_pos = offset;
        for (size_t i = 0; i < shape.size(); i++) {
            out_pos += strides[i] * indices[i];
        }
        out->ptr[out_pos] = val;

        for (int i = shape.size() - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < shape[i]) {
                break;
            }
            indices[i] = 0;
        }
    }
}

// 具体的数学操作可以通过lambda函数传递
void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, std::plus<scalar_t>());
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, std::plus<scalar_t>());
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, std::multiplies<scalar_t>());
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, std::multiplies<scalar_t>());
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, std::divides<scalar_t>());
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, std::divides<scalar_t>());
}

void EwiseEqauls(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, std::equal_to<scalar_t>());
}

void ScalarEqauls(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, std::equal_to<scalar_t>());
}

void EwiseGt(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= b.ptr[i]);
  }
}

void ScalarGt(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= val);
  }
}

void EwiseMax(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMax(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}

// 填充函数保持不变
void Fill(AlignedArray* out, scalar_t val) {
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
    for(size_t i = 0; i < out->size; i++){
      out->ptr[i] = a.ptr[i * reduce_size];
      for(size_t j = 1; j < reduce_size; j++){
        if(a.ptr[i * reduce_size + j] > out->ptr[i]){
          out->ptr[i] = a.ptr[i * reduce_size + j];
        }
      }
    }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
    for(size_t i = 0; i < out->size; i++){
      out->ptr[i] = a.ptr[i * reduce_size];
      for(size_t j = 1; j < reduce_size; j++){
        out->ptr[i] += a.ptr[i * reduce_size + j];
      }
    }
}

void MatMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t shape_a, 
            uint32_t shape_b, uint32_t shape_c){
  for(int i = 0; i < shape_a; i++){
    for(int j = 0; j < shape_c; j++){
      out->ptr[i * shape_c + j] = 0;
      for(int k = 0; k < shape_b; k++){
        out->ptr[i * shape_c + j] += (a.ptr[i * shape_b + k] * b.ptr[k * shape_c + j]);
      }

    }
  }               
}

  
  


}

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });
  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_log", EwiseLog);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("ewise_eq", EwiseEqauls);
  m.def("scalar_eq", ScalarEqauls);
  m.def("scalar_power", ScalarPower);
  m.def("ewise_ge", EwiseGt);
  m.def("scalar_ge", ScalarGt);
  m.def("ewise_maximum", EwiseMax);
  m.def("scalar_maximum", ScalarMax);
  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("matmul", MatMul);
}
}

