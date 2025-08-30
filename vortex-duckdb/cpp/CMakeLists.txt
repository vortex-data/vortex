# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# This CMake project is only used for IDE support.
# See build.rs for the actual build definition.

# If your editor relies on a compilation database to enable LSP functionality,
# you can generate it by running `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` or
# `bear -- cargo build` (https://github.com/rizsotto/Bear).

cmake_minimum_required(VERSION 3.31)
project(vortex)

set(CMAKE_CXX_STANDARD 17)

# Default to debug if build config is not explicitly set.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif ()

# Enable compiler warnings. Unused parameter warnings are disabled as we
# include DuckDB headers with implementations that have unused parameters.
# Further, allow C++20 designator syntax in C++17.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-c++20-designator")

# Include directories
include_directories(include ../duckdb/duckdb-1.3.2/src/include)

# Add library with source files
add_library(vortex
        client_context.cpp
        config.cpp
        data.cpp
        error.cpp
        expr.cpp
        object_cache.cpp
        scalar_function.cpp
        table_filter.cpp
        table_function.cpp
        vector.cpp
)
