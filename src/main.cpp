#include <iostream>

#include "ray_tracer.hpp"
#include "types.hpp"

int main() {
  crt::Scene scene;

  uint2 resolution = make_uint2(512, 512);
  float3 position = make_float3(0);
  float3 view = make_float3(0, 0, 1);
  float3 up = make_float3(0, 1, 0);
  float2 fov = make_float2(45, 45);
  float aperture_radius = 0.04;
  float focal_distance = 4;
  crt::Camera camera(resolution, position, view, up, fov, aperture_radius,
                     focal_distance);

  crt::RayTracer::Parameter parameter(/* max_trace_depth */ 4);
  crt::RayTracer tracer(scene, parameter);

  crt::Image image = tracer.Render(camera);

  return 0;
}