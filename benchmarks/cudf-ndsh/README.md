# cuDF NDS-H: opt-in Vortex build integration

Step 1 of [the POC plan](../../CUDF_POC_PLAN.md). This directory keeps the upstream
cuDF changes reviewable in Vortex without modifying an existing cuDF checkout.
No reader/writer adapter or Vortex benchmark measurements are implemented yet.

## Contents

- `upstream.patch`: adds `CUDF_NDSH_WITH_VORTEX` (default **OFF**), a pinned Vortex
  dependency linked privately to Q1/Q6, an explicit `NDSH_VORTEX_BUILD_SMOKE`
  executable, and `cpp/benchmarks/ndsh/VORTEX.md` with build instructions.
- `test_build_integration.py`: offline configure/build tests for option gating,
  private links, parent settings, excluded targets, and the local-source override.

| Repository        | Pinned revision                            |
| ----------------- | ------------------------------------------ |
| cuDF patch base   | `5339497a1a17d799687cbf189fb113411fb015ca` |
| Vortex dependency | `bffdca1109e99e6957ea2fc18f4a7809c88e0a0c` |

Only the CUDA-enabled FFI archive is used, through `Vortex::cpp_static`. Vortex
examples/tests stay disabled, so they do not fetch another nanoarrow. With the
option off, the integration does not fetch/configure Vortex or discover Rust.

## Apply and build

For a fresh checkout, run from the Vortex repository root:

```sh
git clone https://github.com/NVIDIA/cudf.git build/cudf-ndsh-src
git -C build/cudf-ndsh-src checkout --detach 5339497a1a17d799687cbf189fb113411fb015ca
git -C build/cudf-ndsh-src apply --check ../../benchmarks/cudf-ndsh/upstream.patch
git -C build/cudf-ndsh-src apply ../../benchmarks/cudf-ndsh/upstream.patch
```

The development checkout at `build/cudf-ndsh-src` is already patched; do not apply
it twice. The separate `/home/ubuntu/cudf` checkout was left untouched.

In a RAPIDS development environment, configure only the needed build features:

```sh
cmake -S build/cudf-ndsh-src/cpp -B build/cudf-ndsh-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON \
  -DCUDF_NDSH_WITH_VORTEX=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
CARGO_BUILD_JOBS=4 cmake --build build/cudf-ndsh-build \
  --target NDSH_VORTEX_BUILD_SMOKE --parallel 2
build/cudf-ndsh-build/benchmarks/NDSH_VORTEX_BUILD_SMOKE
CARGO_BUILD_JOBS=4 cmake --build build/cudf-ndsh-build \
  --target NDSH_Q06_NVBENCH NDSH_Q01_NVBENCH --parallel 2
```

Choose the architecture for your GPU (`90` is Hopper). To use local Vortex sources,
add `-DFETCHCONTENT_SOURCE_DIR_VORTEX=/absolute/path/to/vortex`. Otherwise CMake
fetches the pinned complete checkout. Uncached CMake/Cargo dependencies and the
nvCOMP SDK require network access.

Use a consistent compiler/sysroot and CUDA toolkit for cuDF and Vortex. Set
`CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` explicitly when multiple toolkits are
installed. Cargo-owned NVCC also needs `NVCC_CCBIN` when selecting a non-default
host compiler; check for conflicting `NVCC_PREPEND_FLAGS` from Conda activation.
Keep the Cargo build tree intact: Vortex's CUB/nvCOMP runtime libraries are not
packaged by this POC.

## Validation performed

The [supplemental validation record](VALIDATION.md) preserves the exact standalone
consumer setup, compiler/toolkit selections, native relink command, and runtime probe.

On 2026-09-09, Linux aarch64 / GH200 (SM90), CUDA 13.1.115, Conda GCC 14.3,
Rust 1.98.0, and CMake 4.3.2:

- **Passed:** all four offline integration tests and all 13 existing FFI CMake tests.
- **Passed:** full upstream cuDF configuration with benchmarks and Vortex enabled,
  using the local Vortex override and explicit Conda compilers/CUDA toolkit.
- **Passed:** real Vortex CUDA static archive and C++ wrapper compilation with
  Conda GCC 14.3 and CUDA 13.1 in a standalone consumer of prebuilt cuDF.
- **Blocked:** final Conda-toolchain link of that consumer against the existing
  cuDF 26.08 build (`e4b0646588790a00054e4dfa65a4eaa9ba6aa609`). The prebuilt library
  requires glibc 2.32/2.34/2.38 symbols unavailable in the Conda glibc-2.28 sysroot;
  the link also reported unresolved `cudfArrow*` dependency symbols.
- **Passed, supplemental check:** relinking the same smoke objects/archives with
  system GCC 13.3, system runtime libraries, and the prebuilt nanoarrow library's
  link search path; the cuDF/Vortex CUDA initialization executable ran successfully.
  This is not a successful end-to-end Conda CMake build.
- **Passed, limited runtime check:** loaded prebuilt cuDF, the newly compiled CUB
  helper, and Vortex nvCOMP 5.1.0.21 together. CUB and Zstd temporary-size queries
  succeeded while cuDF's nvCOMP 5.2.0.10 was also loaded. This does not validate
  actual decompression or the current upstream nvCOMP 5.3 combination.

Remaining: build/link Q1/Q6 against a fresh cuDF built with the same toolchain,
validate that build's runtime dependencies, and repeat on x86_64. Source fetching
without the local override and relocatable deployment have not been validated.
No SF data generation, reader benchmarks, or profiling was run.

Run the lightweight checks from the Vortex root:

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
python3 -B -m unittest discover -s vortex-ffi/cmake/tests -v
ruff check benchmarks/cudf-ndsh/test_build_integration.py
ruff format --check benchmarks/cudf-ndsh/test_build_integration.py
```

The patched C++ smoke source and CMake module were also checked with cuDF's
`clang-format` and `cmake-format` configurations. No Rust source/API was changed.
