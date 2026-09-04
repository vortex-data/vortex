# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Entry point for the static Vortex C++ wrapper. It consumes the Rust FFI
# archive from vortex-ffi and exports Vortex::cpp_static. It supports standalone
# configuration and add_subdirectory() from the repository root or a parent.

cmake_minimum_required(VERSION 3.28)

# project() is safe inside a parent's add_subdirectory(). A Vortex-owned top
# level, this directory or the repository root, gets developer defaults below;
# an embedding parent keeps its own.
project(VortexCXX DESCRIPTION "Vortex C++ wrapper" LANGUAGES C CXX)
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
# Cargo, CUDA, and sanitizer options are defined by vortex-ffi/CMakeLists.txt.
option(VORTEX_BUILD_TESTING "Build Vortex tests" OFF)
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
    # By path: Catch2 exports its module directory only to the directory that
    # populated it, which may be vortex-ffi/test.
    include("${catch_SOURCE_DIR}/extras/Catch.cmake")

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
