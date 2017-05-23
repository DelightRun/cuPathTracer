#include <iostream>

#include <cuda_runtime.h>

#include <helper_math.h>

#include "path_tracer.hpp"

constexpr float kAperatureRadius = 0.004; // Thin Lens
constexpr size_t kNumScenes = 2;

const std::string kSceneFiles[kNumScenes]{
    "../scenes/scene-1/scene01.obj", "../scenes/scene-2/scene02.obj",
};
const cupt::Camera kCameras[kNumScenes]{
    cupt::Camera(make_uint2(512, 512), make_float3(0, 5, 15), make_float3(0, 0, -1),
                 make_float3(0, 1, 0), kAperatureRadius, 2),
    cupt::Camera(make_uint2(640, 480), make_float3(1, 7, 25), make_float3(0, -1, -25),
                 make_float3(0, 1, 0), kAperatureRadius, 5),
};

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: PathTracer SCENE-ID OUTPUT [SAMPLES] [DEPTH]"
              << std::endl;
    return -1;
  }

  std::cout << "Initializing CUDA Runtime..." << std::flush;
  cudaSetDevice(0);
  std::cout << "Done" << std::endl;

  cupt::PathTracer::Parameter parameter;
  parameter.mc_sample_times = argc > 3 ? atoi(argv[3]) : 1;
  parameter.max_trace_depth = argc > 4 ? atoi(argv[4]) : 1;
  cupt::PathTracer tracer(parameter);

  const size_t id = atoi(argv[1]) - 1;
  assert(id < kNumScenes);

  const std::string& filename = kSceneFiles[id];
  const std::string basedir(filename, 0, filename.rfind("/") + 1);

  std::cout << "Loading Scene..." << std::flush;
  cupt::Scene scene(filename.c_str(), basedir.c_str());
  tracer.SetScene(scene);
  std::cout << "Done" << std::endl;

  std::cout << "Rendering..." << std::flush;
  const cupt::Camera& camera = kCameras[id];
  cupt::Image image = tracer.Render(camera);
  std::cout << "Done" << ::std::endl;

  std::cout << "Saving Result..." << std::flush;
  bool success = image.Save(argv[2]);
  if (success) {
    std::cout << "Success!" << std::endl;
  } else {
    std::cout << "Failed!" << std::endl;
  }

  return 0;
}