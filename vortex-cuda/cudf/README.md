# cuDF C++ environment

Locked libcudf and C++ build/test dependencies for Linux x86-64 and AArch64.
Requires Pixi 0.84.2 and an NVIDIA driver supporting CUDA 13.0.

From the repository root:

```sh
pixi shell --locked --manifest-path vortex-cuda/cudf/pixi.toml -e cpp
```

Run the desired CMake/CTest commands inside that shell. GPU AMI provisioning uses
this same manifest and lockfile; the environment defines no build or test tasks.

After changing dependencies, regenerate the lockfile with Pixi 0.84.2:

```sh
pixi lock --manifest-path vortex-cuda/cudf/pixi.toml
```
