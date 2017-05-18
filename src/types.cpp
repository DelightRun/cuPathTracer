#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "types.hpp"

namespace crt {

bool Image::Save(const char* filename) const {
  assert(pixels.size() != 0);
  assert(resolution.x * resolution.y == pixels.size());

  std::ofstream fout(filename);
  if (!fout) return false;
  fout << "P3\n";
  fout << resolution.x << ' ' << resolution.y << "\n";
  fout << 255 << '\n';
  for (size_t i = 0; i < pixels.size(); i++) {
    fout << (unsigned)pixels[i].x << ' ' << (unsigned)pixels[i].y << ' '
         << (unsigned)pixels[i].z << ' ';
  }
  fout.close();
  return true;
}

bool Scene::Load(const char* filename, const char* mtl_basedir) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename,
                        mtl_basedir)) {
    return false;
  }

  /* Scale Coordinates
  float min[3], max[3];
  for (size_t i = 0; i < 3; i++) {
    min[i] = std::numeric_limits<float>::max();
    max[i] = std::numeric_limits<float>::min();
  }

  for (size_t v = 0; v < attrib.vertices.size(); v++) {
    min[v % 3] = std::min(min[v % 3], attrib.vertices[v]);
    max[v % 3] = std::max(max[v % 3], attrib.vertices[v]);
  }

  float width = max[0] - min[0];
  float height = max[1] - min[1];
  float depth = max[2] - min[2];
  float scale = 2.0 / std::max(width, height);  // 放缩比例
  std::cout << width << "\t" << height << std::endl << scale << std::endl;
  float offset[3];
  offset[0] = (max[0] - min[0]) / 2;
  offset[1] = (max[1] - min[1]) / 2;
  offset[2] = 0;

  for (size_t v = 0; v < attrib.vertices.size(); v++) {
    attrib.vertices[v] =
        (attrib.vertices[v] - min[v % 3] - offset[v % 3]) * scale;
  }*/

  /* Create Scene */
  triangles.clear();

  for (const auto& shape : shapes) {
    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
      Triangle triangle;

      assert(shape.mesh.num_face_vertices[f] == 3);
      for (size_t v = 0; v < 3; v++) {
        tinyobj::index_t idx = shape.mesh.indices[3 * f + v];

        triangle.vertices[v] =
            make_float3(attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]);
        triangle.normal +=
            make_float3(attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]);
      }
      triangle.normal = normalize(triangle.normal / -3);

      size_t m = shape.mesh.material_ids[f];
      const tinyobj::material_t& material = materials[m];

      assert(material.illum == 4); /* TODO Support More Illum Model */
      triangle.material.emitted_color = make_float3(
          material.ambient[0], material.ambient[1], material.ambient[2]);
      triangle.material.diffuse_color = make_float3(
          material.diffuse[0], material.diffuse[1], material.diffuse[2]);
      triangle.material.specular_color = make_float3(
          material.specular[0], material.specular[1], material.specular[2]);
      triangle.material.dissolve = material.dissolve;
      // triangle.material.dissolve = 1;
      triangle.material.ior = material.ior;

      triangles.push_back(triangle);
    }
  }

  return true;
}

}  // namespace crt