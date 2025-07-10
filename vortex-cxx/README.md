# Vortex C++ Bindings

This directory contains C++ bindings for Vortex using the [cxx](https://cxx.rs/) crate. The bindings provide a C++ interface to Vortex file operations, including reading into Arrow Array stream with advanced pushdown support.

## Building

### Requirements

- CMake 3.16 or higher
- C++17 compatible compiler
- Rust toolchain (for building the Rust components)
- vcpkg (for dependency management)

### Managing dependencies

This repo uses VCPKG for dependency management. Enabling VCPKG is very simple: follow
the [installation instructions](https://vcpkg.io/en/getting-started) or just run the following:

```shell
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
export VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Build Steps

```bash
vcpkg install gtest arrow
```
Note: If you want to do your own dependency management, just skip this step. 

Then build the project:

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

We use `.clang-tidy` and `.clang-format` to setup converion. Both are borrowed from DuckDB.

`cppcoreguidelines-avoid-non-const-global-variables` is removed from `.clang-tidy` because GTest violates it.