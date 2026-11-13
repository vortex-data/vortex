# Shared cuDF test environments

Locked for Linux x86-64 and AArch64: `cpp` provides libcudf and build/test tools;
`python` provides Python cuDF. Both share library versions.

## Local use

Requires Pixi 0.84.2, an NVIDIA GPU, and a driver supporting CUDA 13.0.

```sh
pixi run --locked --manifest-path vortex-cuda/cudf/pixi.toml -e cpp test
pixi run --locked --manifest-path vortex-cuda/cudf/pixi.toml -e python test
```

These GPU smoke tests check the environments, not Vortex interop. Missing dependencies
or GPUs fail rather than skip.

## CI reuse

After checkout:

```yaml
- uses: ./.github/actions/setup-cudf
  with:
      environments: cpp python
```

The action installs and caches either or both environments without globally activating them.
Run tests with the commands above.

## Updates

Keep the CUDA compiler and NVRTC on 13.0 for CI driver compatibility. Regenerate the lock
with Pixi 0.84.2:

```sh
pixi lock --manifest-path vortex-cuda/cudf/pixi.toml
```
