# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Entry point for the static Vortex C++ and Rust FFI libraries. It supports both
# standalone configuration and inclusion from a parent via add_subdirectory().

cmake_minimum_required(VERSION 3.28)

# project() is safe inside a parent's add_subdirectory(); PROJECT_IS_TOP_LEVEL
# tells the two cases apart wherever the defaults below differ.
project(VortexCXX DESCRIPTION "Vortex C++ wrapper" LANGUAGES C CXX)

# Vortex does not use C++ modules, so it does not require compiler-specific
# dependency scanners such as clang-scan-deps.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Standalone builds default to Debug; embedded builds keep the parent's choice.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Public options. Embedding parents set these before add_subdirectory().
option(VORTEX_BUILD_TESTING "Build Vortex C++ tests" OFF)
option(VORTEX_BUILD_EXAMPLES "Build Vortex C++ examples" OFF)
option(VORTEX_ENABLE_CUDA "Build the CUDA-enabled Vortex FFI" OFF)
set(VORTEX_CARGO_PROFILE "" CACHE STRING "Override the Cargo profile selected from CMAKE_BUILD_TYPE")

# Sanitizers instrument the Rust archive, its C dependencies, and the C++
# sources together; Configure.cmake enforces the toolchain constraints.
set(VORTEX_SANITIZER "" CACHE STRING "Instrument Vortex and its consumers with asan or tsan")
option(VORTEX_SANITIZE_RUST_STD
    "Rebuild Rust's standard library with sanitizer instrumentation"
    OFF)

# Standalone builds treat warnings as errors; embedded builds do not impose that
# policy on a parent.
option(VORTEX_WARNINGS_AS_ERRORS
    "Treat warnings in Vortex-owned C++ sources as errors"
    "${PROJECT_IS_TOP_LEVEL}")

# Run the private configure-time integration that defines the Cargo build and
# the imported vortex_ffi_static target consumed below.
include(cmake/Configure.cmake)

# The C++ wrapper over the C FFI. Public headers live under include/.
file(GLOB _vortex_cpp_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Consumers inherit the FFI archive, its headers, and its system libraries
# through this target. Hidden visibility keeps wrapper symbols out of a parent
# shared library's exports.
add_library(vortex_cxx STATIC ${_vortex_cpp_sources})
set_target_properties(vortex_cxx PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_features(vortex_cxx PUBLIC cxx_std_20)
target_include_directories(vortex_cxx PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(vortex_cxx PUBLIC vortex_ffi_static)

# Warning policy is private to Vortex-owned sources so embedding this target
# never injects warning flags into consumers.
target_compile_options(vortex_cxx PRIVATE -Wall -Wextra -Wpedantic)
if(VORTEX_WARNINGS_AS_ERRORS)
    target_compile_options(vortex_cxx PRIVATE -Werror)
endif()

# The only supported consumer-facing target name.
add_library(Vortex::cpp_static ALIAS vortex_cxx)

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
    FetchContent_Declare(
        Catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1)
    FetchContent_MakeAvailable(Catch)
    include(Catch)

    # Catch2 enables std::string_view, optional, variant, and byte support from
    # C++17, in headers and compiled sources alike; build it with those on so it
    # matches the C++23 tests that include its headers.
    target_compile_features(Catch2 PRIVATE cxx_std_17)
    target_compile_definitions(Catch2 PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)

    # Embedding parents must also call enable_testing() for ctest to find these.
    enable_testing()
    add_subdirectory(tests)
endif()

if(VORTEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
