# cuPathTracer
A Monte Carlo Path Tracer implemented in CUDA.

## Features

### Current

- [x] Anti-aliasing
- [x] Depth of Field
- [x] Diffusion & Specular Reflection
- [x] Fresnel Equation for Transmission
- [x] Support `obj` & `mtl` Format
- [x] Store Result in `ppm` Format

### TODO

- [ ] Bidirection Tracing
- [ ] Accelerating Intersection using BVH
- [ ] Use `float4` instead of `float3`
- [ ] Memory Access Accelerating via CUDA Constant & Texture Memory 

## System Environment

+ Debian 8 (Jessie) 64bits, *other systems are not tested*
+ GCC 4.9, **GCC 6 is not allowed!!!**
+ CUDA 8.0, *lower versions are not tested*
+ CMake >= 2.8
+ NVIDIA GeForce Titan Black (CUDA capability: 3.5) *lower CUDA capabilities are not tested*

## Project Structure

### Folders

+ `include/` - 3rd-party headers, including [tinyobjloader](https://github.com/syoyo/tinyobjloader) & some helper functions from NVIDIA's offical CUDA examples (with filename starts with "helper_")
+ `src/` - Source code
+ `scenes/` - 2 demo scenes - `scene-1/` for Cornell Box & `scene-2/` for Glossy Plane
+ `results/` - Some demo results

### Source Files

In `src/` directory:

+ `types.hpp` & `types.cpp` implement some essential data structures such as `Ray`, `Triangle`, `Material`, `Camera`, `Scene`, etc.
+ `constants.hpp` defines some useful constant values.
+ `utils.hpp` implements some helper functions.
+ `path_tracer.hpp` & `path_tracer.cu` implement the Monte Carlo Path Tracing algorithm.

**The code style of the source code files above follows [Google C++ Code Style](https://google.github.io/styleguide/cppguide.html)**

## Usage Guide

### Step 1. Create Working Directory

```bash
$ cd /PATH/TO/PROJECT
$ mkdir build
$ cd build
```

### Step 2. Compile

```bash
$ cmake ..
$ make -j8
```

### Step 3. Execute

The demo program needs 5 command-line parameters - **scene id**, **output filename** **samples per pixel**, **maximum tracing bounces** and **gpu id** respectively.

Currently only 3 scenes are supported:

- `1` for scene 1 (Cornell Box)
- `2` for scene 2 (Glossy Planes)
- `3` for my own scene (modified Cornell Box)

Here is an example:

```bash
$ ./PathTracerDemo 1 scene-1.ppm 2000 10    # Create scene 1 on default gpu (gpu 0)
$ ./PathTracerDemo 2 scene-2.ppm 1000 10 1  # Create Scene 2 on gpu 1
$ ./PathTracerDemo 3 scene-3.ppm 10000 20 2 # Create Scene 3 on gpu 2
```

## Results

_Note: Execution time information are measured by `time` command, e,g.:_

```bash
$ time ./PathTracerDemo 1 scene01-10000-ppm 10000 10 0
``` 

### Scene 1 (Cornell Box)

- **Samples per pixel (spp)**: 100, 1000, 10000
- **Maximum tracing bounces**: 10

#### 100 spp, 15 sec
![100 spp, 15 sec](results/scene01/100spp.ppm "100 spp, 15 sec")

#### 1000 spp, 140 sec
![1000 spp, 140 sec](results/scene01/1000spp.ppm "1000 spp, 140 sec")

#### 10000 spp, 23 min
![10000 spp, 23 min](results/scene01/10000spp.ppm "10000 spp, 23 min")

### Scene 2 (Glossy Planes)

- **Samples per pixel (spp)**: 100, 1000, 10000
- **Maximum tracing bounces**: 10

#### 100 spp, 12 sec
![100 spp, 12 sec](results/scene02/100spp.ppm "100 spp, 12 sec")

#### 1000 spp, 125 sec
![1000 spp, 125 sec](results/scene02/1000spp.ppm "1000 spp, 125 sec")

#### 10000 spp, 21 min
![10000 spp, 21 min](results/scene02/10000spp.ppm "10000 spp, 21 min")

### My Own Scene (modified Cornell Box)

- **Samples per pixel (spp)**: 10000
- **Maximum tracing bounces**: 20

_Note: This scene has **Depth of Field** effect._

#### 10000 spp, 45 min
![10000 spp, 45 min](results/scene03/10000spp.ppm "10000 spp, 45 min")

## License

See [LICENSE](LICENSE)