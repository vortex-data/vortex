# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Entry point for the Rust FFI archive. It builds vortex-ffi with Cargo and
# exports Vortex::ffi_static for C consumers; lang/cpp layers the C++ wrapper on
# top. It supports standalone configuration and add_subdirectory() from a parent.

cmake_minimum_required(VERSION 3.28)

# project() is safe inside a parent's add_subdirectory(). A Vortex-owned top
# level, this directory or the repository root, gets developer defaults below;
# an embedding parent keeps its own.
project(VortexFFI DESCRIPTION "Vortex C FFI" LANGUAGES C CXX)
if(CMAKE_PROJECT_NAME MATCHES "^Vortex")
    set(_vortex_top_level ON)
else()
    set(_vortex_top_level OFF)
endif()

# Vortex does not use C++ modules, so it does not require compiler-specific
# dependency scanners such as clang-scan-deps.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# A standalone configure writes compile_commands.json for editor tooling.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(_vortex_top_level AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Public options. Embedding parents set these before add_subdirectory(). The
# testing and examples options are shared with lang/cpp.
option(VORTEX_BUILD_TESTING "Build Vortex tests" OFF)
option(VORTEX_BUILD_EXAMPLES "Build Vortex examples" OFF)
option(VORTEX_ENABLE_CUDA "Build the CUDA-enabled Vortex FFI" OFF)
set(VORTEX_CARGO_PROFILE "" CACHE STRING "Override the Cargo profile selected from CMAKE_BUILD_TYPE")

# Sanitizers instrument the Rust archive, its C dependencies, and every target
# that links Vortex::ffi_static; Configure.cmake enforces the toolchain constraints.
set(VORTEX_SANITIZER "" CACHE STRING "Instrument Vortex and its consumers with asan or tsan")
option(VORTEX_SANITIZE_RUST_STD
    "Rebuild Rust's standard library with sanitizer instrumentation"
    OFF)

option(VORTEX_WARNINGS_AS_ERRORS
    "Treat warnings in Vortex-owned sources as errors"
    "${_vortex_top_level}")

# Define the Cargo build and the imported vortex_ffi_static archive.
include(cmake/Configure.cmake)

# The only supported consumer-facing name for the C API.
add_library(Vortex::ffi_static ALIAS vortex_ffi_static)

# Test and example dependencies are fetched at configure time and never installed.
if(VORTEX_BUILD_TESTING OR VORTEX_BUILD_EXAMPLES)
    include(FetchContent)
    FetchContent_Declare(
        Nanoarrow
        GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow
        GIT_TAG apache-arrow-nanoarrow-0.8.0)
    FetchContent_MakeAvailable(Nanoarrow)

    # arrow-nanoarrow creates a self-referential symlink:
    # python/subprojects/arrow-nanoarrow -> ../..
    if(IS_SYMLINK "${nanoarrow_SOURCE_DIR}/python/subprojects/arrow-nanoarrow")
        file(REMOVE "${nanoarrow_SOURCE_DIR}/python/subprojects/arrow-nanoarrow")
    endif()
endif()

if(VORTEX_BUILD_TESTING)
    # Embedding parents must also call enable_testing() for ctest to find these.
    enable_testing()
    add_subdirectory(test)
endif()

if(VORTEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
