#include "path_tracer.hpp"

#include <device_launch_parameters.h>

#include <helper_math.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include <curand.h>
#include <curand_kernel.h>

#include "constants.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace crt {

namespace {

__device__ inline float3 ComputeReflectionDirection(const float3 normal,
                                                    const float3 incident) {
  /* Compute reflection direction.
   * Refer to "Rui Wang, Lec12 - Ray Tracing, page 11" */
  return normalize(incident - 2.0 * dot(incident, normal) * normal);
}

__device__ inline float3 ComputeTransmissionDirection(const float3 normal,
                                                      const float3 incident,
                                                      float air_ior,
                                                      float material_ior) {
  /* Compute refraction direction according to Snell's Law.
   * Refer to "Rui Wang, Lec12 - Ray Tracing, page 15" */

  float cos_theta_i = dot(normal, incident);

  float eta = 0;
  if (cos_theta_i >= 0) { /* Material -> Air */
    eta = material_ior / air_ior;
  } else { /* Air -> Material */
    eta = air_ior / material_ior;
  }

  float radicand = 1 - (eta * eta) * (1 - (cos_theta_i * cos_theta_i));
  if (radicand < 0) return make_float3(0.0f);  // No Refrection!!!
  float cos_theta_o = sqrt(radicand);

  return normalize((eta * -cos_theta_i + sign(cos_theta_i) * cos_theta_o) *
                       normal +
                   eta * incident);
}

__device__ float3 ComputeRandomCosineWeightedDirection(
    const float3 normal, const float3 incident, curandState* curand_state) {
  /* Compute a random cosine weighted direction in heimsphere */
  float random1;
  random1 = curand_uniform(curand_state);
  float theta = kTwoPi * random1;

  float random2 = curand_uniform(curand_state);
  float cos_phi = sqrt(random2);
  float sin_phi = sqrt(1 - random2);

  /* Choose a axis not near to normal */
  float3 not_normal;
  if (fabs(normal.x) < kSQRTOfOneThird) {
    not_normal = make_float3(1, 0, 0);
  } else if (fabs(normal.y) < kSQRTOfOneThird) {
    not_normal = make_float3(0, 1, 0);
  } else {
    not_normal = make_float3(0, 0, 1);
  }

  float3 x_axis = cross(normal, not_normal);
  float3 y_axis = cross(normal, x_axis);

  return -sign(dot(normal, incident)) *
         normalize((cos(theta) * sin_phi * x_axis) +
                   (sin(theta) * sin_phi * y_axis) + (cos_phi * normal));
}

__device__ ReflectionType RussianRoulette(const float3 normal,
                                          const float3 incident,
                                          const Material& material,
                                          curandState* state) {
  float3 threshold[3];
  threshold[0] = material.diffuse_color;
  threshold[1] = threshold[0] +
                 material.specular_color * (dot(normal, incident) < 0 ? 1 : 0);
  threshold[2] = threshold[1] + (1 - material.dissolve);

  float3 random =
      threshold[2] * make_float3(curand_uniform(state), curand_uniform(state),
                                 curand_uniform(state));
  if (random <= threshold[0]) {
    return ReflectionType::DIFFUSE;
  } else if (random <= threshold[1]) {
    return ReflectionType::SPECULAR;
  } else {
    return ReflectionType::TRANSMISSION;
  }
}

__global__ void InitializationKernal(size_t* indices, curandState* states,
                                     size_t num_pixels, size_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  indices[idx] = idx;
  curand_init(hash(idx) * hash(seed), 0, 0, &states[idx]);
}

__global__ void RayCastFromCameraKernel(const Camera camera, Ray* rays,
                                        size_t num_pixels, unsigned light,
                                        curandState* states) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;
  size_t x = idx % camera.resolution.x;
  size_t y = camera.resolution.y - idx / camera.resolution.x - 1;

  curandState* const curand_state = &states[idx];

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
  float2 jitter = make_float2(curand_uniform(curand_state) - 0.5,
                              curand_uniform(curand_state) - 0.5);
  // 2. compute distances to the center of image plane
  float2 distances = make_float2(make_uint2(x, y)) + jitter;
  distances /= (make_float2(camera.resolution) - 1);
  distances -= 0.5;
  distances *= size;
  // 3. compute point coordinate
  float3 point = center + x_axis * distances.x + y_axis * distances.y;

  // compute origin of the ray
  float3 origin = camera.position;
  if (camera.aperture_radius > kEpsilon) {
    // generate a random point on the aperture
    float angle = kTwoPi * curand_uniform(curand_state);
    float distance =
        camera.aperture_radius * sqrt(curand_uniform(curand_state));

    float2 coord = make_float2(cos(angle) * distance, sin(angle) * distance);

    origin += x_axis * coord.x + y_axis * coord.y;
  }

  rays[idx].origin = origin;
  rays[idx].direction = normalize(point - origin);
  rays[idx].color = make_float3(light);
}

// __global__ void RayTraceKernel(Scene scene, Ray* rays, int num_pixels);
__global__ void PathTraceKernel(const Triangle* triangles,
                                const size_t num_triangles, size_t* indices,
                                Ray* rays, float3* colors,
                                const size_t num_pixels,
                                curandState* curand_states) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  size_t& index = indices[idx];
  if (index == kMaximumSize) return;

  Ray& ray = rays[index];
  curandState* curand_state = &curand_states[index];

  /** Get the nearest intersection */
  size_t intersection_idx = kInvalidIndex;
  float3 weight = make_float3(1e10f); /* p, q, t */

  for (size_t i = 0; i < num_triangles; i++) {
    float3 w = triangles[i].Hit(ray);
    if (w.z > 0 && w.z < weight.z) {
      weight = w;
      intersection_idx = i;
    }
  }

  /** If no intersection, mark as dead ray */
  if (intersection_idx == kInvalidIndex) {
    index = kInvalidIndex;
  } else {
    /** Get secondary ray */
    const Triangle triangle = triangles[intersection_idx];
    const float distance = weight.z;

    ray.origin += distance * ray.direction;
    const float3 normal = triangle.GetNormal(weight.x, weight.y);

    ReflectionType type =
        RussianRoulette(normal, ray.direction, triangle.material, curand_state);
    if (type == ReflectionType::TRANSMISSION) {
      ray.direction = ComputeTransmissionDirection(
          normal, ray.direction, kAirIoR, triangle.material.ior);
    } else if (type == ReflectionType::SPECULAR) {
      ray.color *= triangle.material.specular_color;
      ray.direction = ComputeReflectionDirection(normal, ray.direction);
    } else if (type == ReflectionType::DIFFUSE) {
      colors[index] += (ray.color * triangle.material.emitted_color);

      ray.color *= triangle.material.diffuse_color;
      ray.direction = ComputeRandomCosineWeightedDirection(
          normal, ray.direction, curand_state);
    }
    ray.origin += kRayOriginBias * ray.direction;
  }
}

}  // namespace

Image PathTracer::Render(const Camera& camera) {
  const size_t num_pixels = camera.resolution.x * camera.resolution.y;

  thrust::device_vector<Triangle> triangles(m_scene.triangles);
  Triangle* const triangles_ptr = thrust::raw_pointer_cast(triangles.data());

  thrust::device_vector<float3> colors(num_pixels, make_float3(0));
  float3* const colors_ptr = thrust::raw_pointer_cast(colors.data());

  thrust::device_vector<Ray> rays(num_pixels);
  Ray* const rays_ptr = thrust::raw_pointer_cast(rays.data());

  thrust::device_vector<curandState> curand_states(num_pixels);
  curandState* const curand_states_ptr =
      thrust::raw_pointer_cast(curand_states.data());

  thrust::device_vector<size_t> indices(num_pixels);
  size_t* indices_ptr = thrust::raw_pointer_cast(indices.data());

  for (size_t counter = 0; counter < m_parameter.mc_sample_times; counter++) {
    indices.resize(num_pixels);

    /* Initialization */
    InitializationKernal<<<divUp(num_pixels, kThreadsPerBlock),
                           kThreadsPerBlock>>>(indices_ptr, curand_states_ptr,
                                               num_pixels, counter);

    /* Create rays from camera */
    RayCastFromCameraKernel<<<divUp(num_pixels, kThreadsPerBlock),
                              kThreadsPerBlock>>>(
        camera, rays_ptr, num_pixels, m_scene.light, curand_states_ptr);

    for (size_t depth = 0; depth < m_parameter.max_trace_depth; depth++) {
      // Step 0. Check if over.
      if (indices.size() == 0) break;

      // Step 1. Trace rays to get secondary rays.
      PathTraceKernel<<<divUp(indices.size(), kThreadsPerBlock),
                        kThreadsPerBlock>>>(triangles_ptr, triangles.size(),
                                            indices_ptr, rays_ptr, colors_ptr,
                                            indices.size(), curand_states_ptr);

      // Step 2. Compact rays, remove dead rays.
      thrust::device_vector<size_t>::iterator end = thrust::remove_if(
          indices.begin(), indices.end(), IsUnsignedMinusOne<size_t>());
      indices.resize(end - indices.begin());
    }
  }

  return Image(camera.resolution, thrust::host_vector<float3>(colors),
               // Color2Pixel(10));
               Color2Pixel(m_parameter.mc_sample_times));
}

}  // namespace crt