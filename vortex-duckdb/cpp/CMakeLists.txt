# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# This CMake project is only used for IDE support (CLion, etc.).
# See build.rs for the actual build definition.
# Prerequisites: Run `cargo build -p vortex-duckdb` first to download DuckDB
# source
#
# If your editor relies on a compilation database to enable LSP functionality,
# you can generate it by running `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` or
# `bear -- cargo build` (https://github.com/rizsotto/Bear).
cmake_minimum_required(VERSION 3.22)
project(vortex_duckdb_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to debug if build config is not explicitly set.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Enable compiler warnings (matching build.rs flags).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

# The symlink points to target/duckdb-source-vX.Y.Z which contains duckdb-X.Y.Z/
file(GLOB DUCKDB_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../duckdb/duckdb-*")
if(DUCKDB_DIRS)
    list(GET DUCKDB_DIRS 0 DUCKDB_DIR)
    set(DUCKDB_INCLUDE "${DUCKDB_DIR}/src/include")
    message(STATUS "Found DuckDB include: ${DUCKDB_INCLUDE}")
else()
    message(FATAL_ERROR
        "DuckDB source not found at ../duckdb/duckdb-*\n"
        "Run 'cargo build -p vortex-duckdb' first to download DuckDB"
    )
endif()

include_directories(include ../include)
include_directories(SYSTEM ${DUCKDB_INCLUDE})

# Auto-discover C++ source files
file(GLOB CPP_SOURCES "*.cpp")
add_library(vortex ${CPP_SOURCES})
