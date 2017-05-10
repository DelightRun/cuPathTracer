#include "ray_tracer.cuh"

#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "types.hpp"

namespace crt {

namespace {

constexpr float kEpsilon = 1e-5;
constexpr float kPi = 3.1415926535897932384626422832795028841971;
constexpr float kTwoPi = 2 * kPi;
constexpr float kArcPerAngle = kPi / 180;

enum ReflectionType { DIFFUSE, SPECULAR, REFRACTION };

struct Ray {
  float3 origin;
  float3 direction;

  __host__ __device__ Ray(float3 origin, float3 direction)
      : origin(origin), direction(normalize(direction)) {}
};

typedef Ray* RayPtr;

struct isDeadRay {
  __host__ __device__ inline bool operator()(RayPtr ptr) {
    return ptr == nullptr;
  }
};

__host__ __device__ inline size_t divUp(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ float3 positionAlongRay(const Ray& ray, float distance) {
  return ray.origin + distance * ray.direction;
}

__host__ __device__ float3 computeReflectionDirection(const float3& normal,
                                                      const float3& incident) {
  // refers to "Rui Wang, Lec12 - Ray Tracing, page 11"
  return 2.0 * dot(normal, incident) * normal - incident;
}

__host__ __device__ float3 computeTransmissionDirection(
    const float3& normal, const float3& incident, float refractiveIndexIncident,
    float refractiveIndexTransmitted) {
  // Snell's Law, refe to "Rui Wang, Lec12 - Ray Tracing, page 15"

  // normal & incident should be normalized
  float cos_theta1 = dot(normal, incident);

  float eta = refractiveIndexIncident / refractiveIndexTransmitted;

  float radicand = 1 - (eta * eta) * (1 - (cos_theta1 * cos_theta1));
  if (radicand < 0) return make_float3(0.0f);  // No Refrection!!!
  float cos_theta2 = sqrt(radicand);

  if (cos_theta1 > 0) {  // normal & incident are on same side
    return (eta * cos_theta1 - cos_theta2) * normal - eta * incident;
  } else {  // normal & incident are on opposite sides
    // should reverse cos_theta1 & normal
    return (eta * cos_theta1 + cos_theta2) * normal - eta * incident;
  }
}

__global__ void initializationKernel(Ray* rays, RayPtr* ray_ptrs,
                                     const size_t num_rays) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t idx = x + y;
  if (idx >= num_rays) return;

  ray_ptrs[idx] = rays + idx;
}

__global__ void RayCastFromCameraKernel(const Camera camera, RayPtr* ray_ptrs,
                                        const size_t num_rays) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t idx = x + y;
  if (idx >= num_rays) return;

  thrust::default_random_engine rng(idx);
  thrust::uniform_real_distribution<float> uniform_distribution(0.0, 1.0);

  // compute axis direction
  float3 x_axis = normalize(cross(camera.view, camera.up));
  float3 y_axis = normalize(cross(x_axis, camera.view));

  // compute size and center position of image plane
  // according to focal distance and fov
  float3 center = camera.position + camera.focal_distance * camera.view;
  float2 size =
      2 * make_float2(
              camera.focal_distance * tan((camera.fov.x / 2) * kArcPerAngle),
              camera.focal_distance * tan((camera.fov.y / 2) * kArcPerAngle));

  // compute the jittered point position on image plane
  // 1. generate random jitter offsets(in pixel) for supersample
  float2 jitter = make_float2(uniform_distribution(rng) - 0.5,
                              uniform_distribution(rng) - 0.5);
  // 2. compute distances to the center of image plane
  float2 distances = make_float2(make_uint2(x, y)) + jitter;
  distances /= make_float2(camera.resolution);
  distances -= 0.5;
  distances *= size;
  // 3. compute point coordinate
  float3 point = center + x_axis * distances.x + y_axis * distances.y;

  // compute origin of the ray
  float3 origin = camera.position;
  if (camera.aperture_radius > kEpsilon) {
    // generate a random point on the aperture
    float angle = kTwoPi * uniform_distribution(rng);
    float distance = camera.aperture_radius * sqrt(uniform_distribution(rng));

    float2 coord = make_float2(cos(angle) * distance, sin(angle) * distance);

    origin += x_axis * coord.x + y_axis * coord.y;
  }

  ray_ptrs[idx]->origin = origin;
  ray_ptrs[idx]->direction = normalize(point - origin);
}

// __global__ void RayTraceKernel(Scene scene, Ray* rays, int num_rays);
__global__ void RayTraceKernel(RayPtr* ray_ptrs, size_t num_rays) {
  // 1. intersect, get the nearest
  // 2. if no intersection, dead ray
  // 3. if light source, TODO
  // 4. MC sample from (DIFFUSE, SPECULAR, TRANSMIT) with direction
  // 5. calculate secondary ray according to sampling result
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_rays) return;

  Ray& ray = *ray_ptrs[idx];
  // TODO
}

}  // namespace

constexpr size_t kThreadsPerBlock = 256;

Image RayTracer::Render(const Camera& camera) {
  Image image(camera.resolution);

  size_t num_rays = camera.resolution.x * camera.resolution.y;
  Ray* rays = NULL;
  checkCudaErrors(cudaMalloc((void**)&rays, num_rays * sizeof(Ray)));

  RayPtr* ray_ptrs = NULL;
  checkCudaErrors(cudaMalloc((void**)&rays, num_rays * sizeof(RayPtr)));

  dim3 block_dim(32, kThreadsPerBlock / 32);
  dim3 grid_dim(divUp(camera.resolution.x, blockDim.x),
                divUp(camera.resolution.y, blockDim.y));

  initializationKernel<<<grid_dim, block_dim>>>(rays, ray_ptrs, num_rays);

  RayCastFromCameraKernel<<<grid_dim, block_dim>>>(camera, rays, num_rays);

  // init rays from camera
  for (unsigned depth = 0; depth < m_parameter.max_trace_depth; depth++) {
    // Step 1. trace rays to get secondary rays
    dim3 block_dim(kThreadsPerBlock);
    dim3 grid_dim(divUp(num_rays, blockDim.x));
    RayTraceKernel<<<grid_dim, block_dim>>>(ray_ptrs, num_rays);

    // Step 2. compact rays, remove dead rays
    thrust::device_ptr<RayPtr> dev_ptr(ray_ptrs);
    thrust::device_ptr<RayPtr> end =
        thrust::remove_if(dev_ptr, dev_ptr + num_rays, isDeadRay());
    num_rays = thrust::raw_pointer_cast(end) - ray_ptrs;
  }

  return image;
}

}  // namespace crt