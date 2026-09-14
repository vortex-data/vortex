# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Entry point for the static Vortex C++ wrapper. It consumes the Rust FFI
# archive from vortex-ffi and exports Vortex::cpp_static. It supports standalone
# configuration and add_subdirectory() from the repository root or a parent.

# CMake 3.25 is required for block() and FetchContent SYSTEM.
cmake_minimum_required(VERSION 3.25)

project(VortexCXX DESCRIPTION "Vortex C++ wrapper" LANGUAGES C CXX)

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
option(VORTEX_WARNINGS_AS_ERRORS
    "Treat warnings in Vortex-owned sources as errors"
    "${_vortex_top_level}")

# The Rust FFI archive comes from vortex-ffi. A standalone configure of this
# directory adds it here; the repository root or a parent may already have.
if(NOT TARGET Vortex::ffi_static)
    add_subdirectory(../../vortex-ffi ffi)
endif()

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
target_link_libraries(vortex_cxx PUBLIC Vortex::ffi_static)

# Warning policy is private to Vortex-owned sources so embedding this target
# never injects warning flags into consumers.
target_compile_options(vortex_cxx PRIVATE -Wall -Wextra -Wpedantic)
if(VORTEX_WARNINGS_AS_ERRORS)
    target_compile_options(vortex_cxx PRIVATE -Werror)
endif()

# The only supported consumer-facing name for the C++ API.
add_library(Vortex::cpp_static ALIAS vortex_cxx)

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
    FetchContent_Declare(
        Catch
        SYSTEM
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1)
    FetchContent_MakeAvailable(Catch)
    # By path: Catch2 exports its module directory only to the directory that
    # populated it, which may be vortex-ffi/test.
    include("${catch_SOURCE_DIR}/extras/Catch.cmake")

    # Catch2's headers and library must agree on standard-dependent features.
    target_compile_features(Catch2 PRIVATE cxx_std_20)
    target_compile_definitions(Catch2 PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)

    # Embedding parents must also call enable_testing() for ctest to find these.
    enable_testing()
    add_subdirectory(tests)
endif()

if(VORTEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
