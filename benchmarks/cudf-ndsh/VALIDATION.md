# Validation

2026-09-09, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                             | Result                               |
| ----------------------------------------------------------------- | ------------------------------------ |
| Adapter, matching prebuilt cuDF 26.08 headers/libraries, GCC 13.3 | 12 tests passed                      |
| Same adapter under Compute Sanitizer                              | 0 errors                             |
| Three C++ sources, pinned cuDF headers, Conda GCC 14.3 flags      | Compile-only passed                  |
| Rust decimal / layout / Arrow / CUDA FFI tests                    | 95 / 12 / 221 / 10 passed            |
| Rust doctests                                                     | 74 passed, 21 ignored                |
| NDS-H / existing FFI CMake tests                                  | 7 / 13 passed                        |
| Clippy, Rust/C++/CUDA/CMake formatting, Ruff                      | Passed                               |
| Patch forward/reverse application and whitespace                  | Passed                               |
| Fresh pinned libcudf build                                        | Compiler error; timed out after 600s |

Adapter coverage: 19 column variants, decimals, nulls, typed empty tables,
sliced/small batches, compact string staging, ownership, error recovery, and
async-resource completion on a non-default stream; up to 8,193 rows.
Prerequisite regressions were demonstrated failing before their fixes.

## Runtime setup

Uses cuDF 26.08 at `e4b0646588790a00054e4dfa65a4eaa9ba6aa609` in
`/home/ubuntu/cudf/cpp/build`, its matching nanoarrow, system GCC 13.3, and the
rebuilt combined CUDA-enabled Vortex FFI archive. The host-specific consumer is
`build/cudf-vortex-roundtrip/CMakeLists.txt`; the archive comes from
`build/cudf-vortex-smoke/build/_deps/vortex-build/ffi/vortex-artifacts`.
Rebuild the archive after changing Rust or kernels.

```sh
cmake --build build/cudf-vortex-roundtrip/build --target vortex_io_test --parallel 2
build/cudf-vortex-roundtrip/build/vortex_io_test
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/cudf-vortex-roundtrip/build/vortex_io_test
```

Current-cuDF compile checks used only the three source compilation commands from
`ninja -t commands NDSH_VORTEX_IO_TEST NDSH_VORTEX_BUILD_SMOKE`, with current Vortex
headers. They did not link against the older library: those ABIs differ.

## Rust checks

```sh
cargo +nightly fmt --all
cargo clippy --all-targets --all-features
cargo test -p vortex-array --lib arrays::decimal::
cargo test -p vortex-cuda --lib layout:: -- --test-threads=2
cargo test -p vortex-cuda --lib arrow:: -- --test-threads=2
cargo test -p vortex-cuda-ffi --lib -- --test-threads=2
cargo test --doc -p vortex-array -p vortex-cuda -p vortex-cuda-ffi
```

## Blockers and limits

The fresh `cudf` target build reported `cudf::ast::literal::ast_scalar is private`
in `cudf/detail/utilities/dispatchers.hpp:71`, then timed out. No retry or causal
attribution to this patch. Current pinned linking/runtime and Q1/Q6 builds remain
unverified.

Earlier Conda linking against prebuilt cuDF failed on its glibc-2.28 sysroot;
system relinking passed. Loading Vortex nvCOMP 5.1.0.21 alongside prebuilt cuDF's
5.2.0.10 passed limited API probes, not a decompression compatibility test.
Current cuDF's nvCOMP 5.3.0.16 combination is untested.

No x86_64/deployment checks, SF data, throughput measurements, or Nsight profiles.
The extra bitmap repack and adapter copies must be included in future read timing.
