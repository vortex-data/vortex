# Vortex C++ bindings

Vortex provides a C++20 API for reading and writing Vortex files. See the
[C++ quickstart](../../docs/getting-started/cpp.rst) for API examples.

## Quick start

Build from the repository root with CMake 3.28 or newer, Ninja, native C/C++ compilers, and
Cargo and rustc available to CMake's program search:

```sh
cmake -S lang/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
```

CMake builds the Rust FFI through Cargo. A separate `cargo build` is not required.

## Embed in a CMake project

Vendor or fetch a pinned, complete Vortex checkout, then add `lang/cpp` once:

```cmake
add_subdirectory(path/to/vortex/lang/cpp vortex-cpp)
target_link_libraries(my_target PRIVATE Vortex::cpp_static)
```

`lang/cpp/CMakeLists.txt` is the only supported entry point. Its only supported consumer target is
`Vortex::cpp_static`, an alias of the `vortex_cxx` build target. It transitively links the separate
Rust FFI archive and required native libraries; consume the target rather than copying
`libvortex_cxx.a` alone.

The integration is source-only: it does not install Vortex or provide `find_package(Vortex)`. It
enables C and C++ when needed, but keeps compile and link policy target-scoped and does not replace
the parent's build type, language standards, compiler flags, linker flags, or `BUILD_SHARED_LIBS`.

### Shared-library boundary

Vortex can be linked privately into a shared library such as `libcudf`. Keep Vortex calls in private
C++ implementation files and preserve the parent's symbol-export policy. The parent must prevent
`vx_*` and other implementation symbols from becoming part of its public ABI, for example with an
ELF version script and `--exclude-libs,ALL` or a macOS exported-symbol allowlist. CUDA-enabled builds
also have runtime-library deployment requirements described below.

## Supported configurations

The build supports:

- native GNU/Linux on x86_64 and aarch64;
- native macOS on arm64 for standalone development;
- a single-config generator; unless `VORTEX_CARGO_PROFILE` is set, `Debug`, `Release`,
  `RelWithDebInfo`, and `MinSizeRel` map to Cargo `dev`, `release`, `release_debug`, and
  `release_size`, respectively, while other build types warn and use `dev`; and
- Cargo and rustc 1.95.0 or newer, with the selected target's standard library installed.

CMake discovers Cargo and rustc with `find_program`. Rustup proxies are resolved from the Vortex
workspace, so they honor its `rust-toolchain.toml` and `RUSTUP_TOOLCHAIN`; concrete non-rustup
binaries are used directly. Rustc and the selected C and C++ compilers must target the same native
platform and architecture; the C and C++ compilers must support `-dumpmachine`.

Cross-compilation, Apple universal binaries, Windows, musl, multi-config generators, and shared
Vortex targets are not supported. Ninja is recommended. macOS is not a supported cuDF integration
target.

## Configuration

Set these public Vortex-specific options before `add_subdirectory`, or pass them with `-D` for a
standalone build:

- `VORTEX_BUILD_TESTING=ON` builds the `vortex_cxx_test` C++23 test target. Embedded builds require
  the parent to enable CTest. Default: `OFF`.
- `VORTEX_BUILD_EXAMPLES=ON` builds the examples. Default: `OFF`.
- `VORTEX_ENABLE_CUDA=ON` selects the Linux-only `vortex-cuda-ffi` archive, adds `vortex_cuda.h` to
  the existing `Vortex::cpp_static` target, and requires `find_package(CUDAToolkit)`. Set
  `CUDAToolkit_ROOT` when CMake cannot find the toolkit. Default: `OFF`.
- `VORTEX_CARGO_PROFILE` overrides the Cargo profile inferred from `CMAKE_BUILD_TYPE`. Custom
  profiles use the same-named Cargo artifact directory; Cargo's `test` and `bench` profiles are not
  supported. Default: empty.
- `VORTEX_WARNINGS_AS_ERRORS` promotes warnings only while compiling `vortex_cxx`; it does not affect
  tests, examples, consumers, Rust, or CUDA compilation. Default: `ON` standalone and `OFF` when
  embedded.
- `VORTEX_SANITIZER=asan|tsan` requires `Debug`, Clang or AppleClang, and nightly Rust. ASan enables
  address and undefined-behavior checks for native code and address instrumentation for Rust; TSan
  enables thread instrumentation. Flags propagate to targets linking Vortex, but CUDA device code,
  the CUB helper, and nvCOMP are not sanitizer-instrumented. Default: empty.
- `VORTEX_SANITIZE_RUST_STD=ON` rebuilds Rust's standard library with the selected sanitizer and
  requires the nightly `rust-src` component. Default: `OFF`.

The selected Cargo and rustc binaries are fixed until CMake is reconfigured. Cargo runs with the
lockfile, the selected native target and profile, and the selected FFI package's default features
disabled. Optional features such as `mimalloc` are not enabled. The integration supplies its complete
Rust flag sequence, overriding Rust flags from the environment and Cargo configuration.

Cargo's target cache is stored under `<CMake binary dir>/cargo-target`. The Cargo target runs whenever
Vortex is built, while Cargo decides whether recompilation is needed. The standard CMake clean target
removes both CMake outputs and this Cargo cache:

```sh
cmake --build build/cpp --target clean
```

Public headers are consumed from the source checkout. `vortex.h` and `vortex_cuda.h` are checked in.
A non-sanitizer nightly build may regenerate `vortex.h` with cbindgen and attempt to format it with
`clang-format`; stable and sanitizer builds leave it unchanged. CUDA builds also generate kernel
sources and PTX in the source checkout, so nightly and CUDA builds require it to be writable.

Cargo may download locked dependencies during the build. CUDA builds additionally require libclang
for bindgen and may download the pinned CUDA 12 nvCOMP SDK directly from NVIDIA. Tests and examples
may download Nanoarrow, Catch2, and magic_enum during CMake configure.

### CUDA deployment

CUDA builds continue to use `Vortex::cpp_static`; no separate CUDA CMake target is created. Current
CUDA build scripts invoke NVCC with `-arch=native`, so `CMAKE_CUDA_ARCHITECTURES` has no effect.
Generated PTX is embedded in the Rust archive, but CMake does not stage `libvortex_cub.so` or the
downloaded `libnvcomp.so`. The CUB helper must remain at its original Cargo build path or be placed
next to the host executable; nvCOMP is loaded from its original Cargo build path. CUDA operations
also require a compatible NVIDIA driver and accessible GPU. Consequently, CUDA-enabled output is
not currently a self-contained relocatable deployment artifact.

## Development

### Tests

```sh
cmake -S lang/cpp -B build/cpp-tests -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DVORTEX_BUILD_TESTING=ON
cmake --build build/cpp-tests --parallel
ctest --test-dir build/cpp-tests --output-on-failure
```

### Examples

```sh
cmake -S lang/cpp -B build/cpp-examples -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVORTEX_BUILD_EXAMPLES=ON
cmake --build build/cpp-examples --parallel
```

The `reader`, `writer`, `dtype`, `scan`, and `scan_to_arrow` executables are written to
`build/cpp-examples/examples/`.

### Sanitizers

Install nightly Rust, then select it explicitly. CMake must use Clang or AppleClang:

```sh
rustup toolchain install nightly
RUSTUP_TOOLCHAIN=nightly cmake -S lang/cpp -B build/cpp-asan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DVORTEX_SANITIZER=asan \
    -DVORTEX_BUILD_TESTING=ON
cmake --build build/cpp-asan --parallel
ctest --test-dir build/cpp-asan --output-on-failure
```

To also instrument Rust's standard library, install `rust-src` and configure with
`-DVORTEX_SANITIZE_RUST_STD=ON`.

### Coverage

Run one of the following from `lang/cpp`. The helper reports C++ coverage only and requires a
compatible gcov/LCOV toolchain, plus `genhtml` for HTML output:

```sh
cd lang/cpp
./gcov-report.sh       # Writes coverage.info
# Or:
./gcov-report.sh html  # Also writes coverage/
```
