# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Entry point for the static Vortex C++ and Rust FFI libraries. It supports both
# standalone configuration and inclusion from a parent via add_subdirectory().

cmake_minimum_required(VERSION 3.28)

set(_vortex_workspace_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(_vortex_cmake_dir "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# A standalone configure owns project() and its language set. When embedded,
# preserve the parent's project metadata and enable only languages it has not
# already initialized.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(VortexCXX DESCRIPTION "Vortex C++ wrapper" LANGUAGES C CXX)
    set(_vortex_cpp_standalone ON)
else()
    set(_vortex_cpp_standalone OFF)
    get_property(_vortex_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "C" IN_LIST _vortex_enabled_languages)
        enable_language(C)
    endif()
    if(NOT "CXX" IN_LIST _vortex_enabled_languages)
        enable_language(CXX)
    endif()
endif()

# A toolchain-file change must re-run configure so Cargo and rustc are resolved
# again. Cargo itself tracks manifests, the lockfile, and Rust sources at build
# time because its phony CMake target runs on every build.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_vortex_workspace_root}/rust-toolchain.toml")

option(VORTEX_BUILD_TESTING "Build Vortex C++ tests" OFF)
option(VORTEX_BUILD_EXAMPLES "Build Vortex C++ examples" OFF)
option(VORTEX_ENABLE_CUDA "Build the CUDA-enabled Vortex FFI" OFF)
set(VORTEX_CARGO_PROFILE "" CACHE STRING "Override the Cargo profile selected from CMAKE_BUILD_TYPE")
set(VORTEX_SANITIZER "" CACHE STRING "Instrument Vortex and its consumers with asan or tsan")
set_property(CACHE VORTEX_SANITIZER PROPERTY STRINGS "" asan tsan)

# Standalone builds treat warnings as errors; embedded builds do not impose that
# policy on a parent.
if(_vortex_cpp_standalone)
    set(_vortex_warnings_as_errors_default ON)
else()
    set(_vortex_warnings_as_errors_default OFF)
endif()
option(VORTEX_WARNINGS_AS_ERRORS
    "Treat warnings in Vortex-owned C++ sources as errors"
    "${_vortex_warnings_as_errors_default}")

# Run the private configure-time integration that defines the Cargo build and
# imported Rust FFI target used by the public C++ library.
include("${_vortex_cmake_dir}/Configure.cmake")

set(_vortex_cpp_sources
    src/array.cpp
    src/data_source.cpp
    src/dtype.cpp
    src/expression.cpp
    src/scalar.cpp
    src/scan.cpp
    src/session.cpp
    src/writer.cpp)
add_library(vortex_cxx STATIC ${_vortex_cpp_sources})
set_target_properties(vortex_cxx PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_features(vortex_cxx PUBLIC cxx_std_20)
# Warning policy is private to Vortex-owned sources so embedding this target
# never injects warning flags into consumers.
target_compile_options(vortex_cxx PRIVATE -Wall -Wextra -Wpedantic)
if(VORTEX_WARNINGS_AS_ERRORS)
    target_compile_options(vortex_cxx PRIVATE -Werror)
endif()
target_include_directories(vortex_cxx PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(vortex_cxx PUBLIC vortex_ffi_static)

add_library(Vortex::cpp_static ALIAS vortex_cxx)

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
    target_compile_features(Catch2 PRIVATE cxx_std_17)
    target_compile_definitions(Catch2 PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)

    if(_vortex_cpp_standalone)
        enable_testing()
    endif()
    add_subdirectory(tests)
endif()

if(VORTEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
