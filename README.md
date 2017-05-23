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
+ GCC 4.9, **GCC 6 is not allowed**
+ CUDA 8.0, *lower versions are not tested*
+ CMake >= 2.8

## Demo Guide

### 1. Create Working Directory

```bash
$ cd /PATH/TO/PROJECT
$ mkdir build
$ cd build
```

### 2. Compile

```bash
$ cmake ..
$ make -j8
```

### 3. Usage

The demo program needs 4 command-line parameters - **scene id**, **output filename** **sample time** and **maximum tracing depth** respectively. Currently only 2 scenes are supported - `1` for **Cornell Box** and `2` for **Glossy Plane**.

Here is an example:

```bash
$ ./PathTracerDemo 1 scene-1.ppm 2000 20 # Create Scene 1
$ ./PathTracerDemo 2 scene-2.ppm 1000 10 # Create Scene 2
```

## License

See [LICENSE](LICENSE)