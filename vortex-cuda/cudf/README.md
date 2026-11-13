# cuDF test environment

Locked, standalone dependencies for C++ cuDF tests. This does not affect Cargo or CPU builds.
Supports Linux x86-64 (CI) and AArch64 (local GPU development).

Use Pixi 0.84.2 with an NVIDIA GPU and a driver supporting CUDA 13.0:

```sh
pixi run --locked --manifest-path vortex-cuda/cudf/pixi.toml test
```

The smoke test compiles and runs a CUDA kernel and a mixed-type cuDF binary operation,
including cuDF's test utilities and GTest. A missing GPU or failed kernel is an error, not a skip.
The environment is also available for other tests via `pixi run --locked --manifest-path ...`.

CUDA compiler and NVRTC versions stay on 13.0 to match the GPU CI driver; newer PTX can be
incompatible even within CUDA 13.x. Other runtime libraries follow libcudf's package requirements.
`pixi.lock` pins both platforms. CI uses `--locked` and must not change it. To update deliberately:

```sh
pixi lock --manifest-path vortex-cuda/cudf/pixi.toml
```
