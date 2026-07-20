# Vortex C++ bindings

For a usage guide, see docs/api/cpp/index.rst.

## Requirements

- CMake 3.10+
- C++20 compiler (C++23 compiler for tests).
- Rust toolchain for building `vortex-ffi`.

## Build

```sh
cargo build --release -p vortex-ffi
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This will generate `libvortex_cxx` shared and static libraries.
You can use `vortex_cxx` and `vortex_cxx_shared` CMake targets.

## Test

```sh
cargo build --release -p vortex-ffi
cmake -Bbuild -DBUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build -j "$(nproc)"
```

## Run examples

```sh
cmake -Bbuild -DBUILD_EXAMPLES=ON
cmake --build build -j
./build/examples/hello-vortex
```

## Check coverage

This will generate an LCOV directory `coverage`:

```sh
./gcov-report.sh generate
```
