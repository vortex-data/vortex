# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
cmake_minimum_required(VERSION 3.10)

project(VortexFFI
    VERSION 0.0.1
    LANGUAGES C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wextra -Wpedantic")

option(BUILD_TESTS "Build tests" OFF)

set(SANITIZER "" CACHE STRING "Build with sanitizers")
set(TARGET_TRIPLE "" CACHE STRING "Rust target triple for FFI library")
set(RUST_BUILD_PROFILE "" CACHE STRING "Cargo profile name for the Rust FFI library")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if (NOT SANITIZER STREQUAL "")
    message(NOTICE "Sanitizer: ${SANITIZER}")
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "Only debug build is supported for sanitizer builds")
    endif()

    if (SANITIZER STREQUAL "asan")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address,undefined,leak")
        if (BUILD_TESTS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined,leak")
        endif()
    elseif (SANITIZER STREQUAL "tsan")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
        if (BUILD_TESTS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
        endif()
    else()
        message(FATAL_ERROR "Unknown sanitizer ${SANITIZER}")
    endif()
endif()

message(NOTICE "Build type: ${CMAKE_BUILD_TYPE}")
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND
    NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND
    NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(FATAL_ERROR "vortex-ffi can only be built in Release, Debug, or RelWithDebInfo mode")
endif()

if (RUST_BUILD_PROFILE STREQUAL "")
    if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(RUST_BUILD_PROFILE "release_debug")
    else()
        string(TOLOWER ${CMAKE_BUILD_TYPE} RUST_BUILD_PROFILE)
    endif()
endif()
set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/${TARGET_TRIPLE}/${RUST_BUILD_PROFILE}")

if(WIN32)
    set(LIBRARY_PATH "${LIBRARY_DIR}/libvortex_ffi.lib")
    set(LIBRARY_PATH_SHARED "${LIBRARY_DIR}/libvortex_ffi.dll")
elseif(APPLE)
    set(LIBRARY_PATH "${LIBRARY_DIR}/libvortex_ffi.a")
    set(LIBRARY_PATH_SHARED "${LIBRARY_DIR}/libvortex_ffi.dylib")
else()
    set(LIBRARY_PATH "${LIBRARY_DIR}/libvortex_ffi.a")
    set(LIBRARY_PATH_SHARED "${LIBRARY_DIR}/libvortex_ffi.so")
endif()

set(LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/cinclude")

message(NOTICE "
Library dir ${LIBRARY_DIR}
Shared library path ${LIBRARY_PATH_SHARED}
Static library path ${LIBRARY_PATH}
Headers path ${LIBRARY_HEADERS}")

if (NOT EXISTS "${LIBRARY_PATH_SHARED}")
    message(FATAL_ERROR "Shared library not found")
endif()
if (NOT EXISTS "${LIBRARY_PATH}")
    message(FATAL_ERROR "Static library not found")
endif()

add_library(vortex_ffi STATIC IMPORTED)
set_target_properties(vortex_ffi PROPERTIES
    IMPORTED_LOCATION "${LIBRARY_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBRARY_HEADERS}"
)

add_library(vortex_ffi_shared SHARED IMPORTED)
set_target_properties(vortex_ffi_shared PROPERTIES
    IMPORTED_LOCATION "${LIBRARY_PATH_SHARED}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBRARY_HEADERS}"
    INTERFACE_LINK_OPTIONS "LINKER:-rpath,${LIBRARY_DIR}"
)

if (BUILD_TESTS)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -Wpedantic")

    enable_testing()
    add_subdirectory(test)
endif()
