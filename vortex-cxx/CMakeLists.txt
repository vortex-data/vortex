# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.22)

if(DEFINED ENV{VCPKG_TOOLCHAIN_PATH})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_TOOLCHAIN_PATH}")
else()
    message(WARNING "VCPKG_TOOLCHAIN_PATH environment variable is not set. Dependencies may not be found correctly. See README.md for vcpkg setup instructions.")
endif()

project(vortex-cxx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic -Wno-dollar-in-identifier-extension")

option(VORTEX_ENABLE_TESTING "Enable building test binary for vortex-cxx" OFF)

# Fetch Corrosion
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

# Import Rust crate using Corrosion
corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)

# Enable CXX bridge for the Rust crate
corrosion_add_cxxbridge(vortex_cxx_bridge CRATE vortex_cxx_rs FILES lib.rs)

# Find Arrow
find_package(Arrow CONFIG REQUIRED)

# Collect C++ source files
set(CPP_SOURCE_FILE
    cpp/write_options.cpp
    cpp/file.cpp
    cpp/runtime.cpp
)

# Public headers
set(CPP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

# Create the main library combining C++ and Rust code
add_library(vortex_cxx STATIC ${CPP_SOURCE_FILE})
target_include_directories(vortex_cxx PUBLIC ${CPP_INCLUDE_DIRS})
target_include_directories(vortex_cxx PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/src
)
target_link_libraries(vortex_cxx PUBLIC vortex_cxx_rs vortex_cxx_bridge Arrow::arrow_static)
add_dependencies(vortex_cxx vortex_cxx_rs vortex_cxx_bridge)

# Platform-specific configuration
# TODO(xinyu): test on mac
if(APPLE)
    target_link_libraries(vortex_cxx "-framework CoreFoundation -framework Security")
endif()

# Tests
if (VORTEX_ENABLE_TESTING)
    find_package(GTest CONFIG REQUIRED)

    enable_testing()
    add_executable(vortex_cxx_test cpp/tests/basic_test.cpp)
    target_include_directories(vortex_cxx_test PUBLIC ${CPP_INCLUDE_DIRS})
    target_link_libraries(vortex_cxx_test GTest::gtest_main vortex_cxx Arrow::arrow_static)

    include(GoogleTest)
    gtest_discover_tests(vortex_cxx_test)
endif()