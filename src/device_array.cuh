#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace crt {

// TODO Not Complete

template <typename T>
class DeviceArray {
 public:
  typedef T data_type;
  typedef T& reference_type;
  typedef T* pointer_type;

  __host__ DeviceArray(size_t size) : m_size(size) {
    checkCudaErrors(cudaMalloc((void**)&m_data, m_size * sizeof(T)));
  }

  __host__ ~DeviceArray() {
    if (m_data != nullptr) {
      checkCudaErrors(cudaFree((void*)m_data));
    }
  }

  __host__ __device__ inline size_t size() { return m_size; }
  __host__ __device__ inline pointer_type ptr() { return m_data; }

  __device__ inline reference_type operator[](size_t idx) {
    return m_data[idx];
  }

 private:
  pointer_type m_data;
  size_t m_size;
};

}  // namespace crt

#endif