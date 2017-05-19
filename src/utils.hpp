#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>

#include <helper_math.h>

#include "constants.hpp"

namespace cupt {

inline std::ostream& operator<<(std::ostream& os, float3 value) {
  os << "( " << value.x << ", " << value.y << ", " << value.z << " )";
  return os;
}

__host__ __device__ inline size_t divUp(const size_t a, const size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ inline bool iszerof(const float value) {
  return fabs(value) < kEpsilon;
}

__host__ __device__ inline bool iszero(const float3 value) {
  return iszerof(value.x) && iszerof(value.y) && iszerof(value.z);
}

template <typename T>
__host__ __device__ inline int sign(const T value) {
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}
__host__ __device__ inline float3 pow(const float3 value, const float exp) {
  return make_float3(powf(value.x, exp), powf(value.y, exp),
                     powf(value.z, exp));
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

__host__ __device__ inline bool operator<=(const float3 a, const float3 b) {
  return (a.x <= b.x && a.y <= b.y && a.z <= b.z);
}

struct IsInvalidIndex {
  __host__ __device__ inline bool operator()(const size_t value) const {
    return value == kInvalidIndex;
  }
};

struct Color2Pixel {
  const float normalizer;

  Color2Pixel() : normalizer(1.0) {}
  Color2Pixel(const float normalizer) : normalizer(normalizer) {}

  __host__ __device__ inline uchar3 operator()(float3 color) const {
    float3 c = clamp(pow(color / normalizer, 1 / 2.2), 0, 1) * 255;
    return make_uchar3(c.x, c.y, c.z);
  }
};

}  // namespace cupt

#endif