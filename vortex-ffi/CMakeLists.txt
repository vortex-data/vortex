# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Entry point for the Rust FFI archive. It builds vortex-ffi with Cargo and
# exports Vortex::ffi_static for C consumers; lang/cpp layers the C++ wrapper on
# top. It supports standalone configuration and add_subdirectory() from a parent.

# CMake 3.25 is required for block() and FetchContent SYSTEM.
cmake_minimum_required(VERSION 3.25)

project(VortexFFI DESCRIPTION "Vortex C FFI" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Standalone entry points own their defaults; otherwise inherit Vortex's marker.
if(PROJECT_IS_TOP_LEVEL)
    set(_vortex_top_level ON)
elseif(NOT DEFINED _vortex_top_level)
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

option(VORTEX_BUILD_TESTS "Build Vortex tests" OFF)
option(VORTEX_BUILD_EXAMPLES "Build Vortex examples" OFF)
option(VORTEX_ENABLE_CUDA "Build the CUDA-enabled Vortex FFI" OFF)
set(VORTEX_CARGO_PROFILE "" CACHE STRING "Override the Cargo profile selected from CMAKE_BUILD_TYPE")

# Sanitizers instrument the Rust archive, its C dependencies, and every target
# that links Vortex::ffi_static; Configure.cmake enforces the toolchain constraints.
set(VORTEX_SANITIZER "" CACHE STRING
    "Comma-separated sanitizers for Vortex and its consumers: asan, lsan, ubsan, tsan")
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
if(VORTEX_BUILD_TESTS OR VORTEX_BUILD_EXAMPLES)
    include(FetchContent)

    # Keep dependency warnings out of Vortex's warning policy.
    set(NANOARROW_DEBUG_EXTRA_WARNINGS OFF)
    FetchContent_Declare(
        Nanoarrow
        SYSTEM
        GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow
        GIT_TAG apache-arrow-nanoarrow-0.8.0)
    FetchContent_MakeAvailable(Nanoarrow)
    # Suppress Nanoarrow warnings.
    target_compile_options(nanoarrow_static PRIVATE -w)
    target_compile_options(nanoarrow_shared PRIVATE -w)

    # arrow-nanoarrow creates a self-referential symlink:
    # python/subprojects/arrow-nanoarrow -> ../..
    if(IS_SYMLINK "${nanoarrow_SOURCE_DIR}/python/subprojects/arrow-nanoarrow")
        file(REMOVE "${nanoarrow_SOURCE_DIR}/python/subprojects/arrow-nanoarrow")
    endif()
endif()

if(VORTEX_BUILD_TESTS)
    # Embedding parents must also call enable_testing() for ctest to find these.
    enable_testing()
    add_subdirectory(test)
endif()

if(VORTEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
