#ifndef RAY_TRACER_CUH
#define RAY_TRACER_CUH

#include "types.hpp"

namespace cupt {

class PathTracer {
 public:
  struct Parameter {
    // unsigned char device;
    unsigned int max_trace_depth;
    unsigned int mc_sample_times;
  };

  PathTracer(const Parameter& parameter) : m_parameter(parameter) {}
  PathTracer(const Scene& scene, const Parameter& parameter)
      : m_scene(scene), m_parameter(parameter) {}

  void SetScene(const Scene& scene) { m_scene = scene; }

  Image Render(const Camera&);

 private:
  Parameter m_parameter;
  Scene m_scene;
};

}  // namespace cupt

#endif