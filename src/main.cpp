#include <iostream>

#include <cuda_runtime.h>

#include <helper_math.h>

#include "path_tracer.hpp"

crt::Scene DummyScene() {
  crt::Triangle triangle;
  triangle.normal = make_float3(0, 0, -1);
  triangle.vertices[0] = make_float3(1, 1, 1);
  triangle.vertices[1] = make_float3(-1, 1, 1);
  triangle.vertices[2] = make_float3(1, -0.5, 1);
  triangle.material.emitted_color = make_float3(0.78, 0, 0.78);

  crt::Scene scene;
  scene.triangles.push_back(triangle);
  return scene;
}

crt::Camera DummyCamera() {
  uint2 resolution = make_uint2(1024, 1024);
  float3 position = make_float3(0);
  float3 view = make_float3(0, 0, 1);
  float3 up = make_float3(0, 1, 0);
  float2 fov = make_float2(90, 90);
  float aperture_radius = 0.04;
  float focal_distance = 4;

  return crt::Camera(resolution, position, view, up, fov, aperture_radius,
                     focal_distance);
}

int main() {
  std::cout << "Initializing CUDA Runtime..." << std::flush;
  cudaSetDevice(0);
  std::cout << "Done" << std::endl;

  crt::PathTracer::Parameter parameter;
  parameter.max_trace_depth = 5;
  parameter.mc_sample_times = 50;
  crt::PathTracer tracer(parameter);

  tracer.SetScene(DummyScene());

  std::cout << "Rendering" << std::endl;
  crt::Image image = tracer.Render(DummyCamera());
  std::cout << "Done" << ::std::endl;

  std::cout << "Saving Result..." << std::flush;
  bool success = image.Save("./result.ppm");
  if (success) {
    std::cout << "Success!" << std::endl;
  } else {
    std::cout << "Failed!" << std::endl;
  }

  return 0;
}