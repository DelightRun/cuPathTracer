#include "path_tracer.hpp"

#include <helper_math.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include <curand_kernel.h>

#include "constants.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace cupt {

namespace {

__constant__ Camera camera;

__device__ inline float3 ComputeReflectionDirection(const float3 normal,
                                                    const float3 incident) {
  /* Compute reflection direction.
   * Refer to "Rui Wang, Lec12 - Ray Tracing, page 11" */
  return normalize(incident - 2.0 * dot(incident, normal) * normal);
}

__device__ float3 ComputeRandomCosineWeightedDirection(
    const float3 normal, const float3 incident, const float shininess,
    curandState* curand_state) {
  /* Compute a random cosine weighted direction in heimsphere */
  float random1 = curand_uniform(curand_state);
  float cos_phi = powf(random1, 1.0 / (1 + shininess));
  float sin_phi = sqrt(1 - square(cos_phi));

  float random2 = curand_uniform(curand_state);
  float theta = kTwoPi * random2;

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

  float3 direction = ((cos(theta) * sin_phi * x_axis) +
                      (sin(theta) * sin_phi * y_axis) + (cos_phi * normal));
  return normalize(direction);
}

__device__ inline float3 ComputeTransmissionDirection(
    const float3 normal, const float3 incident, const float eta,
    curandState* /* ignored */) {
  /* Compute refraction direction according to Snell's Law.
   * Refer to "Rui Wang, Lec12 - Ray Tracing, page 15" */
  float cos_theta_i = dot(normal, incident);

  float radicand = 1 - square(eta) * (1 - square(cos_theta_i));
  if (radicand < 0) /* Total Internal Reflection */
    return make_float3(0);
  float cos_theta_o = sqrt(radicand);

  float3 direction =
      (sign(cos_theta_i) * cos_theta_o - eta * cos_theta_i) * normal +
      eta * incident;
  return normalize(direction);
}

__device__ float3 ComputeSpecularityDirection(const float3 normal,
                                              const float3 incident,
                                              const float shininess,
                                              curandState* curand_state) {
  float3 perfect = ComputeReflectionDirection(normal, incident);
  return ComputeRandomCosineWeightedDirection(perfect, incident, shininess,
                                              curand_state);
}

__device__ float3 ComputeDiffusionDirection(const float3 normal,
                                            const float3 incident,
                                            curandState* curand_state) {
  return ComputeRandomCosineWeightedDirection(normal, incident, 1.0,
                                              curand_state);
}

__global__ void InitializationKernal(size_t* indices,
                                     curandState* curand_states,
                                     size_t num_pixels, size_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  indices[idx] = idx;
  curand_init(hash(idx) * hash(seed), 0, 0, &curand_states[idx]);
}

__global__ void RayCastFromCameraKernel(Ray* rays, const size_t num_pixels,
                                        const float intensity,
                                        curandState* curand_states) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  size_t x = idx % camera.resolution.x;
  size_t y = camera.resolution.y - idx / camera.resolution.x - 1;

  curandState* const curand_state = &curand_states[idx];

  /* compute axis direction */
  float3 x_axis = normalize(cross(camera.view, camera.up));
  float3 y_axis = normalize(cross(x_axis, camera.view));

  /* compute image plane ratio and center position */
  float ratio = camera.resolution.x * 1.0 / camera.resolution.y;
  float3 center = camera.position + camera.view * camera.focal_distance;

  /* compute the jittered point position on image plane */
  float2 jitter = make_float2(curand_uniform(curand_state) - 0.5,
                              curand_uniform(curand_state) - 0.5);
  float2 distances = (make_float2(make_uint2(x, y)) + jitter) /
                     (make_float2(camera.resolution) - 1);
  distances = (2 * distances - 1) * make_float2(ratio, 1);
  float3 point = center + distances.x * x_axis + distances.y * y_axis;

  /* compute origin of the ray */
  float3 origin = camera.position;
  if (camera.aperture_radius > kEpsilon) {
    float angle = kTwoPi * curand_uniform(curand_state);
    float distance =
        camera.aperture_radius * sqrt(curand_uniform(curand_state));
    float2 coord = make_float2(cos(angle) * distance, sin(angle) * distance);
    origin += x_axis * coord.x + y_axis * coord.y;
  }

  rays[idx].origin = origin;
  rays[idx].direction = normalize(point - origin);
  rays[idx].color = make_float3(intensity);
}

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
  } else { /** Else get secondary ray */
    /* Transmit ray to the intersection point */
    ray.origin += weight.z * ray.direction;

    const Triangle triangle = triangles[intersection_idx];
    float3 normal = triangle.GetNormal(weight.x, weight.y);
    float shininess = triangle.material.shininess;
    float eta = 1.0;
    bool into = (dot(normal, ray.direction) < 0);
    if (!into) normal *= -1;

    /* Specular material by default */
    float3 diffusion = triangle.material.diffuse_color;
    float3 specularity = triangle.material.specular_color;
    float3 transmission = make_float3(0);

    /* Transparent material, calculate fresnel cofficient */
    if (triangle.material.dissolve < 1) {
      float incident_ior = kAirIoR, transmitted_ior = kAirIoR;
      if (into) /* Air -> Material */
        transmitted_ior = triangle.material.ior;
      else /* Material -> ir */
        incident_ior = triangle.material.ior;
      eta = incident_ior / transmitted_ior;
      shininess = 1000;  // Mirror reflection

      const float3 direction = ComputeTransmissionDirection(
          normal, ray.direction, eta, curand_state);

      if (iszero(direction)) { /* Total Internal Reflection */
        specularity = make_float3(1);
        transmission = make_float3(0);
      } else { /* Calculate Fresnel Cofficient */
        float cos_theta_i = fabs(dot(normal, ray.direction));
        float cos_theta_o = fabs(dot(normal, direction));
        float rs = square(
            (incident_ior * cos_theta_i - transmitted_ior * cos_theta_o) /
            (incident_ior * cos_theta_i + transmitted_ior * cos_theta_o));
        float rt = square(
            (incident_ior * cos_theta_o - transmitted_ior * cos_theta_i) /
            (incident_ior * cos_theta_o + transmitted_ior * cos_theta_i));
        float r = (rs + rt) / 2;

        specularity = make_float3(r);
        transmission = make_float3(1 - r);
      }
    }

    /* Russian Roulette */
    float3 threshold[3];
    threshold[0] = diffusion;
    threshold[1] = threshold[0] + specularity;
    threshold[2] = threshold[1] + transmission;
    float3 random = threshold[2] * make_float3(curand_uniform(curand_state),
                                               curand_uniform(curand_state),
                                               curand_uniform(curand_state));

    if (random <= threshold[0]) { /* Diffusion */
      colors[index] += (ray.color * triangle.material.emitted_color);
      ray.color *= triangle.material.diffuse_color;
      ray.direction =
          ComputeDiffusionDirection(normal, ray.direction, curand_state);
    } else if (random <= threshold[1]) { /* Specular */
      ray.color *= triangle.material.specular_color;
      ray.direction = ComputeSpecularityDirection(normal, ray.direction,
                                                  shininess, curand_state);
    } else if (random <= threshold[2]) { /* Transmission */
      ray.color *= (1 - triangle.material.dissolve);
      ray.direction = ComputeTransmissionDirection(normal, ray.direction, eta,
                                                   curand_state);
    }
    ray.origin += kRayOriginBias * ray.direction;
  }
}

}  // namespace

Image PathTracer::Render(const Camera& host_camera) {
  checkCudaErrors(cudaMemcpyToSymbol(camera, &host_camera, sizeof(Camera)));

  const size_t num_pixels = host_camera.resolution.x * host_camera.resolution.y;

  thrust::device_vector<Triangle> triangles(m_scene.triangles);
  Triangle* const triangles_ptr = thrust::raw_pointer_cast(triangles.data());

  thrust::device_vector<float3> colors(num_pixels, make_float3(0));
  float3* const colors_ptr = thrust::raw_pointer_cast(colors.data());

  thrust::device_vector<Ray> rays(num_pixels);
  Ray* const rays_ptr = thrust::raw_pointer_cast(rays.data());

  thrust::device_vector<curandState> curand_curand_states(num_pixels);
  curandState* const curand_curand_states_ptr =
      thrust::raw_pointer_cast(curand_curand_states.data());

  thrust::device_vector<size_t> indices(num_pixels);
  size_t* indices_ptr = thrust::raw_pointer_cast(indices.data());

  for (size_t counter = 0; counter < m_parameter.mc_sample_times; counter++) {
    /* Initialization */
    indices.resize(num_pixels);
    InitializationKernal<<<divUp(num_pixels, kThreadsPerBlock),
                           kThreadsPerBlock>>>(
        indices_ptr, curand_curand_states_ptr, num_pixels, counter);

    /* Create rays from camera */
    RayCastFromCameraKernel<<<divUp(num_pixels, kThreadsPerBlock),
                              kThreadsPerBlock>>>(
        rays_ptr, num_pixels, m_scene.intensity, curand_curand_states_ptr);

    for (size_t depth = 0; depth < m_parameter.max_trace_depth; depth++) {
      /* Step 0. Check if over. */
      if (indices.size() == 0) break;

      /* Step 1. Trace rays to get secondary rays. */
      PathTraceKernel<<<divUp(indices.size(), kThreadsPerBlock),
                        kThreadsPerBlock>>>(
          triangles_ptr, triangles.size(), indices_ptr, rays_ptr, colors_ptr,
          indices.size(), curand_curand_states_ptr);

      /* Step 2. Compact rays, remove dead rays. */
      thrust::device_vector<size_t>::iterator end =
          thrust::remove_if(indices.begin(), indices.end(), IsInvalidIndex());
      indices.resize(end - indices.begin());
    }
  }

  return Image(host_camera.resolution, colors,
               Color2Pixel(m_parameter.mc_sample_times));
}

}  // namespace cupt