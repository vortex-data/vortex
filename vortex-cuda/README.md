# vortex-cuda

`vortex-cuda` provides CUDA execution and Arrow C Device export support for Vortex arrays.

## Arrow Device export

Key files:

- `vortex-cuda/src/arrow/mod.rs`: Arrow C Device ABI, export traits, and lifetime management.
- `vortex-cuda/src/arrow/canonical.rs`: canonical-array export to `ArrowDeviceArray`.
- `vortex-test/e2e-cuda/src/lib.rs`: cuDF interop harness.

Current export coverage includes primitive, bool, decimal/temporal, string/binary view, and struct arrays. Remaining work includes null masks, broader dtype coverage, `ArrowDeviceArrayStream`, and PyVortex integration.

## cuDF compatibility

Vortex exports string and binary columns as Arrow `Utf8View` / `BinaryView` device arrays with producer-owned `ArrowArray.private_data`. cuDF string/binary interop requires a build containing `rapidsai/cudf#22620`; until a release version is identified, test with a cuDF commit that includes that change.

## Building cuDF for interop testing

Pass a single CUDA architecture, e.g. `-DCMAKE_CUDA_ARCHITECTURES=90a`; otherwise cuDF builds for many architectures and local builds are much slower.

```sh
export PATH=/usr/local/cuda-13.1/bin:$PATH

cmake -S cpp -B cpp/build \
  -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX:-/usr/local} \
  -DCMAKE_CUDA_ARCHITECTURES=90a \
  -DBUILD_TESTS=ON \
  -DDISABLE_DEPRECATION_WARNINGS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUDF_BUILD_STREAMS_TEST_UTIL=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -GNinja

cmake --build cpp/build --target INTEROP_TEST -j$(nproc)

LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH ./cpp/build/gtests/INTEROP_TEST
```

Adjust architecture, compiler paths, and CUDA paths for the machine under test.
