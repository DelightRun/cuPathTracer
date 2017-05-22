cmake_minimum_required(VERSION 2.8)
project(cuPathTracer)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set(INCLUDE_DIR ${cuPathTracer_SOURCE_DIR}/include)
set(SOURCE_DIR ${cuPathTracer_SOURCE_DIR}/src)

include_directories(${cuPathTracer_SOURCE_DIR}/include)

set(PATH_TRACER_FILES ${SOURCE_DIR}/path_tracer.cu ${SOURCE_DIR}/types.cpp)

cuda_add_executable(PathTracerDemo ${SOURCE_DIR}/main.cpp ${COMMON_FILES} ${PATH_TRACER_FILES})