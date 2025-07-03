# This CMake project is only used for IDE support.
# See build.rs for the actual build definition.

# If your editor relies on a compilation database to enable LSP functionality,
# you can generate it by running `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` or
# `bear -- cargo build` (https://github.com/rizsotto/Bear).

cmake_minimum_required(VERSION 3.31)
project(vortex)

set(CMAKE_CXX_STANDARD 17)

# Allow C++20 designator syntax in C++17
add_compile_options(-Wno-c++20-designator)

# Include directories
include_directories(./include)
include_directories(../duckdb/src/include)

# Add library with source files
add_library(vortex
        data.cpp
        error.cpp
        expr.cpp
        scalar_function.cpp
        table_filter.cpp
        table_function.cpp
        vector.cpp
)
