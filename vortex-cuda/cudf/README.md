# Shared cuDF test environments

Locked dependencies for Linux x86-64 (CI) and AArch64 (local GPU development):

- `cpp`: libcudf, test utilities, GTest, CMake/Ninja, and C++/CUDA compilers.
- `python`: Python and cuDF, without adding the C++ build tools.

Both environments share a solve group, keeping common libraries on identical versions.
They do not affect Cargo, CPU builds, or the repository's Python workspace.

## Local use

Use Pixi 0.84.2 with an NVIDIA GPU and a driver supporting CUDA 13.0:

```sh
pixi run --locked --manifest-path vortex-cuda/cudf/pixi.toml -e cpp test
pixi run --locked --manifest-path vortex-cuda/cudf/pixi.toml -e python test
```

The C++ smoke tests run a compiled CUDA kernel and a mixed-type cuDF operation. The Python
smoke tests check numeric/string operations and null handling on real GPU-backed cuDF objects.
Missing dependencies or a missing GPU fail rather than skip. These are environment checks,
not substitutes for Vortex's interop tests.

## CI reuse

After checkout, install either environment or both:

```yaml
- uses: ./.github/actions/setup-cudf
  with:
      environments: cpp python
```

The action pins Pixi, enforces the lockfile, and caches installations without changing the job's
Python or CUDA environment globally. Run commands explicitly with `pixi run --locked ... -e cpp`
or `-e python`. Changes to the action, manifest, lockfile, or smoke tests trigger the GPU workflow.

## Updates

CUDA compiler and NVRTC stay on 13.0 to match the GPU CI driver; newer PTX can be incompatible
even within CUDA 13.x. Other runtime libraries follow libcudf's package requirements.
CI must not change `pixi.lock`. Update it deliberately with:

```sh
pixi lock --manifest-path vortex-cuda/cudf/pixi.toml
```
