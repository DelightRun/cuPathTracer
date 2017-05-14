#ifndef UTILS_HPP
#define UTILS_HPP

#include <helper_math.h>

#include <iostream>

#include "constants.hpp"

namespace crt {

template <typename T>
struct IsUnsignedMinusOne {
  __host__ __device__ inline bool operator()(const T value) const {
    return value == (T)-1;
  }
};

struct Color2Pixel {
  const float normalizer;

  Color2Pixel() : normalizer(1.0) {}
  Color2Pixel(const float normalizer) : normalizer(normalizer) {}

  __host__ __device__ inline uchar3 operator()(const float3 color) const {
    const float3 c = clamp((color / normalizer) * 255, 0, 255);
    return make_uchar3(c.x, c.y, c.z);
  }
};

__host__ __device__ inline size_t divUp(const size_t a, const size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ inline bool iszero(const float value) {
  return fabs(value) < kEpsilon;
}

template <typename T>
__host__ __device__ inline int sign(const T value) {
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

__host__ __device__ inline unsigned hash(const unsigned value) {
  unsigned a = value;
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc261c23c) ^ (a >> 19);
  a = (a + 0x166667b1) + (a << 5);
  a = (a + 0xd352646c) ^ (a << 9);
  a = (a + 0xfd4046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__host__ __device__ inline bool operator<(const float3 a, const float3 b) {
  return (a.x < b.x && a.y < b.y && a.z < b.z);
}

}  // namespace crt

#endif