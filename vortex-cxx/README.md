# Vortex C++ Bindings

This directory contains C++ bindings for Vortex using the [cxx](https://cxx.rs/) crate. The bindings provide a C++ interface to Vortex file operations, including roundtripping with Arrow Array stream with advanced pushdown support.

## Building

### Requirements

- CMake 3.22 or higher
- C++17 compatible compiler
- Rust toolchain (for building the Rust components)

### Build Steps

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
# Enable tests in CMake
cmake -DVORTEX_ENABLE_TESTING=ON ..
make -j$(nproc)
./vortex_cxx_test
```

## C++ Coding Convention

We use `.clang-tidy` and `.clang-format` to setup the coding convention. Both are borrowed from DuckDB.

`cppcoreguidelines-avoid-non-const-global-variables` is removed from `.clang-tidy` because GTest violates it.