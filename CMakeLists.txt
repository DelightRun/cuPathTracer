cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_35,code=sm_35")

include_directories(include)

set(COMMON_FILES types.hpp)
set(RAY_TRACER_FILES ray_tracer.hpp ray_tracer.cu)

cuda_add_executable(RayTracerDemo main.cpp ${COMMON_FILES} ${RAY_TRACER_FILES})