# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.22)

project(vortex-examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_custom_target(convert-parquet
    COMMAND cargo run -p vortex-tui convert ${CMAKE_CURRENT_SOURCE_DIR}/goldenfiles/example.parquet
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Converting example.parquet using vortex-tui"
    VERBATIM
)

add_dependencies(hello-vortex convert-parquet)
