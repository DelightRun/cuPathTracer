# cuPathTracer
A Monte Carlo Path Tracer implemented in CUDA.

## Features

- [x] Anti-aliasing
- [x] Depth of Field
- [x] Diffusion & Specular Reflection
- [x] Fresnel Equation for Transmission
- [x] Support `obj` & `mtl` Format
- [x] Store Result in `ppm` Format
- [ ] Bidirection Tracing
- [ ] Bounding Volume Hierarchy

## System Environment

+ Debian 8 (Jessie) 64bits, *other systems are not tested*
+ GCC 4.9, **GCC 6 is not allowed!!!**
+ CUDA 8.0, *lower versions are not tested*
+ CMake >= 2.8
+ NVIDIA GeForce Titan Black & Titan X **Computability >= 3.5 required!!!**

## Project Structure

### Folders

+ `include/` - 3rd-party headers, including [tinyobjloader](https://github.com/syoyo/tinyobjloader) & some helper functions from NVIDIA's offical CUDA examples (with filename starts with "helper_")
+ `src/` - Source code
+ `scenes/` - 2 demo scenes - `scene-1/` for Cornell Box & `scene-2/` for Glossy Plane

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

The demo program needs 5 command-line parameters - **scene id**, **output filename** **sample time**, **maximum tracing depth** and **gpu id** respectively.
Currently only 2 scenes are supported - `1` for **Cornell Box** and `2` for **Glossy Plane**.

Here is an example:

```bash
$ ./PathTracerDemo 1 scene-1.ppm 2000 20   # Create scene 1 on default gpu (gpu 0)
$ ./PathTracerDemo 2 scene-2.ppm 1000 10 1 # Create Scene 2 on gpu 1
```

## Results

## License

See [LICENSE](LICENSE)