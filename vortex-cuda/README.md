# vortex-cuda

`vortex-cuda` provides CUDA execution and Arrow C Device export support for Vortex arrays.

## Arrow Device export

Key files:

- `vortex-cuda/src/arrow/mod.rs`: Arrow C Device ABI, export traits, and lifetime management.
- `vortex-cuda/src/arrow/canonical.rs`: canonical-array export to `ArrowDeviceArray`.
- `vortex-test/e2e-cuda/src/lib.rs`: cuDF interop harness.

## Building cuDF for Arrow Device interop

The `cudf-test-harness` repository provides prebuilt cuDF binaries for Arrow Device interop testing on
x86_64 and aarch64.

From the cuDF repository root, compile the Arrow Device interop target locally without exporting additional
environment variables:

```sh
cmake -E rm -rf cpp/build

cmake -S cpp -B cpp/build \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
  -DBUILD_TESTS=ON \
  -DDISABLE_DEPRECATION_WARNINGS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUDF_BUILD_STATIC_DEPS=OFF \
  -DCUDF_BUILD_STREAMS_TEST_UTIL=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  # In large AArch64 Debug links, CALL26/JUMP26 relocations can exceed their branch range.
  # Use 1 MiB stub groups so GNU ld emits veneers close enough to each relocation site.
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--stub-group-size=1048576" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--stub-group-size=1048576" \
  -GNinja && cmake --build cpp/build --target INTEROP_TEST --parallel
```

## Running the cuDF test harness

```sh
cargo build -p vortex-test-e2e-cuda
cmake --build /path/to/cudf-test-harness/build --target cudf-test-harness --parallel

/path/to/cudf-test-harness/build/cudf-test-harness \
  check-stream \
  target/debug/libvortex_test_e2e_cuda.so
```

To run both `check` and `check-stream` under `compute-sanitizer` for all primitive dtypes:

```sh
target/debug/cudf_harness_runner \
  /path/to/cudf-test-harness/build/cudf-test-harness \
  target/debug/libvortex_test_e2e_cuda.so
```

If cuDF fails with `cudaErrorInsufficientDriver` when using CUDA 13, use the compatibility driver
libraries:

```sh
LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat \
  target/debug/cudf_harness_runner \
  /path/to/cudf-test-harness/build/cudf-test-harness \
  target/debug/libvortex_test_e2e_cuda.so
```
