# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
cmake_minimum_required(VERSION 3.10)

project(VortexFFI
    VERSION 0.0.1
    LANGUAGES C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -Wpedantic")

option(BUILD_TESTING "Build tests" ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(NOTICE "Build type: ${CMAKE_BUILD_TYPE}")
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND
    NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND
    NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(FATAL_ERROR "vortex-ffi can only be built in Release, Debug, or RelWithDebInfo mode")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/release_debug")
else()
    string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
    set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/${CMAKE_BUILD_TYPE_LOWER}")
endif()

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

if (BUILD_TESTING)
    enable_language(CXX)
    enable_testing()
    add_subdirectory(test)
endif()
