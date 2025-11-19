# PointCloud Generator

pointcloud_generator is part of the vk system.
It provides tools for generating and processing 3D point clouds.
All third-party dependencies (e.g., Open3D, Eigen, spdlog, zstd, lz4, etc.) are prebuilt and installed under the ../cache/ directory.

---

## Build Instructions

### 1. Enter the project directory
```bash
cd pointcloud_generator
```
### 2.Unzip Prebuild Thirdparites into cache folder
Download from release
```bash
unzip cache.zip
```

### 3. Configure the project with CMake
```bash
mkdir build && cd build
cmake ..
```

### 4. Build Project
```bash
make
```

## Prebuilt Dependencies

The `cache/` directory contains prebuilt third-party dependencies required by the project.  
These binaries were **precompiled on 6.8.0-85-generic #85~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep 19 16:18:59 UTC 2 x86_64 x86_64 x86_64 GNU/Linux**.

### Contents of `cache/`:
capnp-install  
cereal-install  
CLI11-install  
dai-install  
eigen3-install  
lz4-install  
magic_enum-install  
mcap-include  
Open3D-install  
opengv-install  
rerun-install  
spdlog-install  
zstd-install

## vk_sdk Version

The project uses **vk_sdk version `7f0d0465`**,  
prebuilt and tested on **Ubuntu 22.04 (x86_64)**.