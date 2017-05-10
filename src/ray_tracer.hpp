#ifndef RAY_TRACER_CUH
#define RAY_TRACER_CUH

#include "types.hpp"

namespace crt {

class RayTracer {
 public:
  struct Parameter {
    unsigned char device;
    unsigned int max_trace_depth;

    Parameter(unsigned int max_trace_depth)
        : device(0), max_trace_depth(max_trace_depth) {}
    Parameter(unsigned char device, unsigned int max_trace_depth)
        : device(device), max_trace_depth(max_trace_depth) {}
  };

  RayTracer(const Scene& scene, const Parameter& parameter)
      : m_scene(scene), m_parameter(parameter) {}

  Image Render(const Camera&);

 private:
  Parameter m_parameter;
  Scene m_scene;
};

}  // namespace crt

#endif