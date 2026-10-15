# vortex-cuda

`vortex-cuda` provides CUDA execution and Arrow C Device export support for Vortex arrays.

## Arrow Device export

Key files:

- `vortex-cuda/src/arrow/mod.rs`: Arrow C Device ABI, export traits, and lifetime management.
- `vortex-cuda/src/arrow/canonical.rs`: canonical-array export to `ArrowDeviceArray`.
- `vortex-test/e2e-cuda/src/lib.rs`: cuDF interop harness.

## CUDA file scans

`CudaFileScanExt::scan_cuda(DictionaryExport::Decode)` builds a scan-local reader tree for
plain Arrow Device output. With a CUDA session already installed, CUDA-flat readers eagerly
decode numeric field packs before joining field inputs. Mixed/string projections and raw
primitive auxiliary arrays, such as list offsets, remain lazy.

Use the same dictionary policy in the Arrow export context. The scan does not change the
shared session or ordinary reader cache: `VortexFile::scan()` and
`scan_cuda(DictionaryExport::Preserve)` retain their existing dictionary-preserving behavior.
No CUDA session is initialized by constructing a scan.

Local pooled file reads use fixed 8 MiB chunks with concurrency 32. These are physical
file-read/H2D chunks, independent of logical row ranges or scan batch sizes. File transfers
use one session-scoped H2D stream, separate from the execution pool (four streams by default).
A reader keeps its allocation and chunk writes on the same H2D lane; device-buffer events
order consumers on execution streams. Pinned source buffers stay fenced until DMA completion.
Stream pools initialize each slot once, even under concurrent first use, while keeping the
initialized access path lock-free.

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

LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat \
  target/debug/cudf_harness_runner \
  /path/to/cudf-test-harness/build/cudf-test-harness \
  target/debug/libvortex_test_e2e_cuda.so
```
