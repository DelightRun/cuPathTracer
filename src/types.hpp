#ifndef TYPES_HPP
#define TYPES_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include <curand_kernel.h>

#include "constants.hpp"
#include "utils.hpp"

namespace cupt {

enum TransmissionType { DIFFUSION, REFLECTION, REFRACTION };

struct Ray {
  float3 origin;
  float3 direction;
  float3 color;

  __host__ __device__ Ray()
      : origin(make_float3(0)),
        direction(make_float3(0)),
        color(make_float3(1)) {}
  __host__ __device__ Ray(float3 origin, float3 direction)
      : Ray(origin, direction, make_float3(1)) {}
  __host__ __device__ Ray(float3 origin, float3 direction, float3 color)
      : origin(origin), direction(normalize(direction)), color(color) {}
};

struct Material {
  float3 diffuse_color;
  float3 specular_color;
  float3 emitted_color;

  float dissolve;
  float ior;  // index of refraction

  Material()
      : diffuse_color(make_float3(0)),
        specular_color(make_float3(0)),
        emitted_color(make_float3(0)),
        dissolve(1),
        ior(1) {}

  __host__ __device__ inline bool Emit() const {
    return !iszero(emitted_color);
  }
};

struct Triangle {
  float3 vertices[3];
  float3 normals[3];

  Material material;

  Triangle() {
    for (size_t i = 0; i < 3; i++) {
      vertices[i] = make_float3(0);
      normals[i] = make_float3(0);
    }
  }

  /** \brief Intersect with a given ray using Moller-Trumbore Algorithm
   *  \return distance between ray's origin and intersection point, negative
   * for disjoint
   */
  __host__ __device__ inline float3 Hit(const Ray ray) const {
    float3 edge1 = vertices[1] - vertices[0];
    float3 edge2 = vertices[2] - vertices[0];

    float3 pvec = cross(ray.direction, edge2);

    float det = dot(pvec, edge1);
    if (iszerof(det)) return make_float3(-1);
    float inv_det = 1.0f / det;

    float3 tvec = ray.origin - vertices[0];
    float p = dot(tvec, pvec) * inv_det;

    float3 qvec = cross(tvec, edge1);
    float q = dot(ray.direction, qvec) * inv_det;

    float t = dot(edge2, qvec) * inv_det;

    if (p >= 0.0f && q >= 0.0f && (p + q) <= 1.0f)
      return make_float3(p, q, t);
    else
      return make_float3(-1);
  }

  __host__ __device__ inline float3 GetNormal(const float u,
                                              const float v) const {
    float3 normal = (1 - u - v) * normals[0] + u * normals[1] + v * normals[2];
    return normalize(normal);
  }
};

struct Camera {
  /** \brief Camera resolution. Unit: pixel */
  uint2 resolution;
  /** \brief Camera position. */
  float3 position;
  /** \brief Camera view direction. MUST BE NORMALIZED! */
  float3 view;
  /** \brief Camera up direction, MUST BE NORMALIZED! */
  float3 up;
  /** \brief Aperture radius. */
  float aperture_radius;
  /** \breief Focal distance/length. */
  float focal_distance;

  Camera() {}

  Camera(uint2 resolution, float3 position, float3 view, float3 up,
         float aperture_radius, float focal_distance)
      : resolution(resolution),
        position(position),
        view(normalize(view)),
        up(normalize(up)),
        aperture_radius(aperture_radius),
        focal_distance(focal_distance) {}
};

struct Image {
  uint2 resolution;
  thrust::host_vector<uchar3> pixels;

  Image(const uint2 resolution, thrust::host_vector<float3> colors,
        Color2Pixel color2pixel)
      : resolution(resolution) {
    assert(resolution.x * resolution.y == colors.size());
    pixels.resize(colors.size());
    for (size_t i = 0; i < colors.size(); i++) {
      pixels[i] = color2pixel(colors[i]);
    }
  }

  Image(const uint2 resolution, thrust::device_vector<float3> colors,
        Color2Pixel color2pixel)
      : resolution(resolution) {
    assert(resolution.x * resolution.y == colors.size());
    thrust::host_vector<float3> hcolors(colors);
    pixels.resize(colors.size());
    for (size_t i = 0; i < colors.size(); i++) {
      pixels[i] = color2pixel(hcolors[i]);
    }
  }

  bool Save(const char* filename) const;
};

struct Scene {
  thrust::host_vector<Triangle> triangles;
  unsigned intensity;

  Scene() : intensity(kDefaultIntensity) {}
  Scene(const char* filename, const char* mtl_basedir = NULL) : intensity(kDefaultIntensity) {
    if (!Load(filename, mtl_basedir)) {
      throw "Cannot load scene from file!";
    }
  }

  bool Load(const char* filename, const char* mtl_basedir = NULL);
};

}  // namespace cupt

#endif