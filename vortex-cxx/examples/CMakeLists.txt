# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.22)

project(vortex-examples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(SCCACHE_PROGRAM sccache)
if (SCCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    message(STATUS "Sccache found: ${SCCACHE_PROGRAM}")
else ()
    message(STATUS "Sccache not found")
endif ()

include(FetchContent)
FetchContent_Declare(
    vortex
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../
    SOURCE_SUBDIR vortex-cxx
)
FetchContent_MakeAvailable(vortex)

add_executable(hello-vortex hello-vortex.cpp)

if(APPLE)
    set(APPLE_LINK_FLAGS "-framework CoreFoundation -framework Security")
endif()

target_link_libraries(hello-vortex PRIVATE vortex ${APPLE_LINK_FLAGS})
