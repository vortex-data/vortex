# This CMake project is only used for IDE support. See build.rs for compile commands.

cmake_minimum_required(VERSION 3.31)
project(vortex)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(./include)
include_directories(../../duckdb-vortex/duckdb/src/include/)

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