# Vortex C++ bindings

A C++20 API for reading and writing Vortex files. See the
[C++ quickstart](../../docs/getting-started/cpp.rst) for API examples.

## Build

Requires CMake 3.25+, C/C++ compilers, and Cargo/rustc 1.95+ on `PATH`.

From a Vortex checkout:

```sh
cmake -S lang/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
```

CMake runs Cargo for you. Configuration and builds download uncached dependencies.

**Native builds only:** GNU/Linux x86_64 and aarch64, plus macOS arm64 for standalone development.
Cross-compilation, universal binaries, Windows, musl, and shared Vortex targets are unsupported.

## Embed in a CMake project

Vendor or fetch a pinned, complete checkout:

```cmake
add_subdirectory(path/to/vortex vortex)
target_link_libraries(my_cpp_target PRIVATE Vortex::cpp_static)
target_link_libraries(my_c_target PRIVATE Vortex::ffi_static)
```

Vortex leaves parent build settings unchanged. Its archives are position-independent.

## Build options

Pass `-D<OPTION>=<VALUE>` or set options before `add_subdirectory`. Both bindings share these
options. Defaults below are for standalone builds.

| Option                      | Default  | Purpose                                                     |
| --------------------------- | -------- | ----------------------------------------------------------- |
| `VORTEX_BUILD_TESTS`        | `OFF`    | C API and C++23 wrapper tests.                              |
| `VORTEX_BUILD_EXAMPLES`     | `OFF`    | C/C++ examples.                                             |
| `VORTEX_WARNINGS_AS_ERRORS` | `ON`     | Warnings as errors for Vortex targets only.                 |
| `VORTEX_CARGO_PROFILE`      | Inferred | Override the mapping below.                                 |
| `VORTEX_RUSTUP_TOOLCHAIN`   | Inferred | [Rust toolchain override](#toolchain-and-build-behavior).   |
| `VORTEX_SANITIZER`          | Empty    | [Sanitizers](#sanitizers): `asan`, `lsan`, `ubsan`, `tsan`. |
| `VORTEX_SANITIZE_RUST_STD`  | `OFF`    | Also instrument Rust's standard library.                    |
| `VORTEX_DEBUG_INFO`         | `2`      | C/C++ and Rust debug info: `0` none, `1` limited, `2` full. |
| `VORTEX_ENABLE_CUDA`        | `OFF`    | Linux-only [CUDA build](#cuda).                             |

Embedded builds default `VORTEX_WARNINGS_AS_ERRORS` to `OFF`. Parents must call
`enable_testing()` to register tests.

### Cargo profiles

Standalone builds default to `Debug`. Unless overridden, Cargo uses:

| `CMAKE_BUILD_TYPE` | Cargo profile         |
| ------------------ | --------------------- |
| `Debug` or empty   | `dev`                 |
| `Release`          | `release`             |
| `RelWithDebInfo`   | `release_debug`       |
| `MinSizeRel`       | `release_size`        |
| Other              | `dev`, with a warning |

Cargo's `test` and `bench` profiles are unsupported.

### Toolchain and build behavior

- On first configuration, `VORTEX_RUSTUP_TOOLCHAIN` defaults to `RUSTUP_TOOLCHAIN`, otherwise
  `nightly` for Rust sanitizers or the workspace `rust-toolchain.toml`. CMake caches this choice.
  Set it explicitly to change toolchains. An empty value selects the workspace toolchain.
- Cargo/rustc are found on `PATH`. Override them with `VORTEX_CARGO_EXECUTABLE` and
  `VORTEX_RUSTC_EXECUTABLE`. CMake replaces environment/configuration Rust flags and builds with
  the lockfile and no optional FFI features.
- Cargo's native dependencies use CMake's compilers, archiver, SDK, and C/C++ flags, except
  warning-as-error flags. Global flags can override dependency choices, including language standards.
  Prefer target-scoped flags or Vortex options.
- Cargo caches under the FFI build directory (`ffi/cargo-target` in C++/root builds).
  `clean` removes this cache and staged headers, not checkout headers.
- Nightly builds without Rust sanitizers may regenerate `vortex.h`.
  Nightly and CUDA builds need a writable checkout for generated sources.

## Development

### Tests and examples

```sh
cmake -S lang/cpp -B build/cpp-dev -DVORTEX_BUILD_TESTS=ON -DVORTEX_BUILD_EXAMPLES=ON
cmake --build build/cpp-dev --parallel
ctest --test-dir build/cpp-dev --output-on-failure
```

Examples appear in `build/cpp-dev/examples/` (C++) and `build/cpp-dev/ffi/examples/` (C).

Build-system regressions require Python 3.11+, CMake, Ninja, Make, Rust, Clang, and `nm`.
They compile a tiny Cargo/C/C++ fixture, not Vortex:

```sh
cargo fetch --locked
python3 -m unittest discover -s vortex-ffi/cmake/tests -v
```

Missing tools fail the tests rather than skipping them. These checks cover embedding,
compiler flags, rebuilds, generated headers, and CUDA architecture forwarding;
the C/C++ CI jobs build and test Vortex itself.

### Sanitizers

Use `Debug`. `asan`, `lsan`, and `tsan` require nightly Rust and upstream LLVM Clang.
`ubsan` is native-only and also supports stable Rust and AppleClang.

```sh
rustup toolchain install nightly

cmake -S lang/cpp -B build/cpp-asan \
    -DCMAKE_BUILD_TYPE=Debug -DVORTEX_RUSTUP_TOOLCHAIN=nightly \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DVORTEX_SANITIZER=asan,ubsan -DVORTEX_BUILD_TESTS=ON
cmake --build build/cpp-asan --parallel
ctest --test-dir build/cpp-asan --output-on-failure
```

Set the nightly explicitly when reusing a stable build tree. To instrument Rust's standard library,
install `rust-src` for that nightly and add `-DVORTEX_SANITIZE_RUST_STD=ON` alongside a Rust sanitizer.

Flags instrument Vortex, targets linking it, and Cargo's native library dependencies—not Cargo's
build tools or FetchContent dependencies. CUDA device code, the CUB helper, and nvCOMP are excluded.

### Coverage

Requires gcov/LCOV and, for HTML output, `genhtml`. Reports C++ coverage only.

```sh
cd lang/cpp
./gcov-report.sh html
```

Omit `html` for `coverage.info` only. `CMAKE_BUILD_PARALLEL_LEVEL` overrides the online CPU count.
Coverage instrumentation stays off Cargo's build tools.

## CUDA

`VORTEX_ENABLE_CUDA=ON` selects `vortex-cuda-ffi` and adds `vortex_cuda.h` to the existing targets.
Requires Linux, a CUDA toolkit (`CUDAToolkit_ROOT` if needed), and libclang.
The build downloads the pinned CUDA 12 nvCOMP SDK if uncached.

`CMAKE_CUDA_ARCHITECTURES` controls both Vortex kernels and CUB. For example:

```sh
cmake -S lang/cpp -B build/cpp-cuda -DVORTEX_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80-real;90-virtual"
```

- `80-real` emits only machine code for `sm_80`.
- `80-virtual` emits only PTX with compute capability 8.0 as its baseline (`compute_80`).
  The driver JIT-compiles it for GPUs with compute capability 8.0 or newer, not older GPUs.
- `80` emits both machine code and PTX. The driver uses compatible machine code when available,
  otherwise it JIT-compiles the PTX automatically.

The `80` is a GPU capability baseline, not a PTX ISA version. The driver must support the PTX
ISA version emitted by the CUDA toolkit. Architecture-specific (`a`) and family-specific (`f`)
PTX targets have narrower compatibility than ordinary numeric targets.

`native` (the default) targets the GPUs visible at build time without PTX fallback.
`all` and `all-major` use NVCC's corresponding architecture selection; `OFF` leaves selection
to NVCC's defaults. Explicit targets allow GPU-less builds; GPU and toolkit compatibility
still apply. With `native`, clean the build cache when switching GPU hosts.
This does not retarget the prebuilt nvCOMP SDK.

**Deployment limits:**

- Kernel sources are generated in the checkout. Compiled kernels live in Cargo's build directory
  and are embedded in the archive as fat binaries.
- CMake does not stage shared libraries. `libvortex_cub.so` must remain at its Cargo build path
  or beside the executable. `libnvcomp.so` is loaded from its original Cargo build path.
- CUDA operations require a compatible NVIDIA driver and an accessible GPU.

End-to-end cuDF integration, GCC 14, Conda compiler wrappers, and glibc 2.28 remain unvalidated.
