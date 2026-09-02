# Vortex C++ bindings

Vortex provides a C++20 API for reading and writing Vortex files. See the
[C++ quickstart](../../docs/getting-started/cpp.rst) for API examples.

## Quick start

Build from the repository root with CMake 3.28 or newer, Ninja, native C/C++ compilers, and the
workspace Rust toolchain:

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

`lang/cpp/CMakeLists.txt` is the only supported entry point. It provides one public target:
`Vortex::cpp_static`, a static PIC C++20 library with all required transitive dependencies.

The integration is source-only: it does not install Vortex or provide `find_package(Vortex)`. It
enables C and C++ when needed, but keeps compile and link policy target-scoped and does not replace
the parent's build type, language standards, compiler flags, linker flags, or `BUILD_SHARED_LIBS`.

### Shared-library boundary

Vortex can be linked privately into a shared library such as `libcudf`. Keep Vortex calls in private
C++ implementation files and preserve the parent's symbol-export policy. The parent must prevent
`vx_*` and other implementation symbols from becoming part of its public ABI, for example with an
ELF version script and `--exclude-libs,ALL` or a macOS exported-symbol allowlist.

## Supported configurations

The build supports:

- native GNU/Linux on x86_64 and aarch64;
- native macOS on x86_64 and arm64 for standalone development;
- an empty `CMAKE_BUILD_TYPE` (Rust development profile), `Debug`, `Release`, and `RelWithDebInfo`; and
- Cargo and rustc 1.95.0 or newer, with the selected target's standard library installed.

Cargo, rustc, and the CMake-selected C/C++ compilers must target the same native platform and
architecture. The workspace toolchain is used by default.

Cross-compilation, Apple universal binaries, Windows, musl, multi-config generators, and shared
Vortex targets are not supported. macOS is not a supported cuDF integration target.

## Configuration

Set options before `add_subdirectory`, or pass them with `-D` for a standalone build.

### Build options

- `VORTEX_BUILD_TESTING=ON` builds the C++23 tests. Embedded builds require the parent to enable
  CTest. Default: `OFF`.
- `VORTEX_BUILD_EXAMPLES=ON` builds the examples. Default: `OFF`.
- `VORTEX_WARNINGS_AS_ERRORS` applies only to Vortex-owned C++ sources. Default: `ON` standalone and
  `OFF` when embedded.
- `VORTEX_SANITIZER=asan|tsan` enables sanitizer instrumentation. Use an explicit Debug build and a
  nightly Rust toolchain with `rust-src`. Flags propagate to targets that link Vortex, and `mimalloc`
  cannot be used with sanitizers.

### Cargo options

- `VORTEX_CARGO_EXECUTABLE` and `VORTEX_RUSTC_EXECUTABLE` override tool discovery.
- `VORTEX_CARGO_TARGET_DIR` selects the Cargo cache root.
- `VORTEX_CARGO_JOBS` sets Cargo's job limit.
- `VORTEX_CARGO_OFFLINE=ON` passes `--offline` to Cargo; the Cargo cache must already be complete.
- `VORTEX_CARGO_FEATURES=mimalloc` enables the only supported optional FFI feature.
- `VORTEX_RUSTFLAGS` adds rustc arguments. `target-cpu=native` is rejected because parent artifacts
  may run on a different CPU.

CMake resolves and validates the toolchain during configure. The selected Cargo and rustc binaries
remain fixed until the project is reconfigured, and CMake forwards its
native toolchain settings to
Cargo build scripts.

Cargo artifacts are isolated by a fingerprint of the toolchain and build settings. CMake clean
removes the staged Vortex archive but preserves the Cargo cache. Ambient Rust flag variables are
ignored; use `VORTEX_RUSTFLAGS` for intentional additions.

Stable builds use the checked-in `vortex-ffi/cinclude/vortex.h`. A non-sanitizer nightly build may
regenerate that header in the source checkout and run `clang-format`; sanitizer builds skip header
generation.

Cargo may download locked Rust dependencies during the build. Tests and examples may download their
CMake `FetchContent` dependencies during configure; `VORTEX_CARGO_OFFLINE` does not affect those
downloads.

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

The executables are written to `build/cpp-examples/examples/`.

### Sanitizers

Install nightly Rust with `rust-src`, then select it explicitly:

```sh
rustup toolchain install nightly --component rust-src
RUSTUP_TOOLCHAIN=nightly cmake -S lang/cpp -B build/cpp-asan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DVORTEX_SANITIZER=asan \
    -DVORTEX_BUILD_TESTING=ON
cmake --build build/cpp-asan --parallel
ctest --test-dir build/cpp-asan --output-on-failure
```

### Coverage

Run the coverage helper from `lang/cpp`. It requires LCOV's `geninfo`, plus `genhtml` for HTML output:

```sh
cd lang/cpp
./gcov-report.sh
./gcov-report.sh html  # Also writes coverage/
```
