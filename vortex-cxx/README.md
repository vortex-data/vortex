# Vortex C++ Bindings

This directory contains C++ bindings for Vortex using the [cxx](https://cxx.rs/) crate. The bindings provide a C++ interface to Vortex arrays, data types, scalar values, and file operations.

## Features

- Safe C++ interface to Vortex arrays
- Type-safe data type handling
- Scalar value access and conversions
- Array slicing and manipulation
- Exception-based error handling
- **File reading capabilities** - Read Vortex files from disk
- **Zero-copy Arrow conversion** - Convert Vortex arrays to native Apache Arrow C++ format using Arrow C ABI

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

Those dependencies are required for the tests.

Then build the project:

```bash
mkdir build
cd build
cmake ..
make
```

### Running Tests

```bash
# Enable tests in CMake
cmake -DVORTEX_ENABLE_TESTING=ON ..
make
ctest
```

## Usage Example

```cpp
#include "vortex.hpp"

int main() {
    // Create a new array
    vortex::Array array;
    
    // Get array length
    std::cout << "Array length: " << array.len() << std::endl;
    
    // Access elements
    auto scalar = array.scalar_at(0);
    vortex::Scalar s(std::move(scalar));
    
    if (!s.is_null()) {
        std::cout << "First element: " << s.as_i32() << std::endl;
    }
    
    // Slice array
    auto sliced = array.slice(0, 2);
    std::cout << "Sliced array length: " << sliced.len() << std::endl;
    
    // Convert to native Arrow C++ (zero-copy using Arrow C ABI)
    auto native_arrow = array.to_arrow();
    std::cout << "Arrow array length: " << native_arrow.length() << std::endl;
    std::cout << "Arrow array format: " << native_arrow.format() << std::endl;
    
    // Read from file (if file exists)
    try {
        auto file = vortex::File::open("data.vortex");
        std::cout << "File rows: " << file.row_count() << std::endl;
        auto file_array = file.read_all();
        std::cout << "File data length: " << file_array.len() << std::endl;
    } catch (const vortex::VortexException& e) {
        std::cout << "File not found or error: " << e.what() << std::endl;
    }
    
    return 0;
}
```

## Error Handling

The C++ bindings use exception-based error handling. Operations that can fail will throw a `vortex::VortexException` with a descriptive error message.

## Architecture

The bindings are structured as follows:

- `src/lib.rs` - Main Rust FFI bridge using cxx, with Arrow C ABI export functionality
- `include/vortex.hpp` - C++ header with high-level API including native Arrow C ABI integration
- `src/*.cpp` - Additional C++ implementation files
- `tests/` - C++ unit tests using GoogleTest, including zero-copy Arrow conversion tests
- `CMakeLists.txt` - CMake build configuration
- `build.rs` - Rust build script for cxx integration