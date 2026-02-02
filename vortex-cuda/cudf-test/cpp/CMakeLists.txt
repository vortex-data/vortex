# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(cudf_arrow_ffi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find RAPIDS dependencies
find_package(cudf REQUIRED)
find_package(rmm REQUIRED)

# Create the shared library
add_library(cudf_arrow_ffi SHARED
    cudf_arrow_ffi.cpp
)

target_include_directories(cudf_arrow_ffi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudf_arrow_ffi
    PUBLIC
        cudf::cudf
        rmm::rmm
)

# Set output directory to parent directory for easier linking from Rust
set_target_properties(cudf_arrow_ffi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Install the library
install(TARGETS cudf_arrow_ffi
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES cudf_arrow_ffi.h
    DESTINATION include
)
