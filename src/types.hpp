#ifndef TYPES_HPP
#define TYPES_HPP

#include <helper_math.h>

namespace crt {

struct Material {
  float3 diffuse_color;
  float3 emitted_color;
  float3 specular_color;

  // TODO tranmission and refraction

  Material(float3 diffuse_color, float3 emitted_color, float3 specular_color)
      : diffuse_color(diffuse_color),
        emitted_color(emitted_color),
        specular_color(specular_color) {}
};

struct Triangle {
  float3 position;
  float3 vertices[3];
  float3 normal;

  Material material;
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
  /** \brief Field Of View */
  float2 fov;

  /** \brief Aperture radius. */
  float aperture_radius;
  /** \breief Focal distance/length. */
  float focal_distance;

  Camera(uint2 resolution, float3 position, float3 view, float3 up, float2 fov,
         float aperture_radius, float focal_distance)
      : resolution(resolution),
        position(position),
        view(normalize(view)),
        up(normalize(up)),
        fov(fov),
        aperture_radius(aperture_radius),
        focal_distance(focal_distance) {}
};

struct Scene {};

struct Image {
  uint2 resolution;

  Image(uint2 resolution) : resolution(resolution) {}
};

}  // namespace crt

#endif