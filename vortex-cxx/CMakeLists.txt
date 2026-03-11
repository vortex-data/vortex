# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.22)

# FetchContent from URL timestamp handling
cmake_policy(SET CMP0135 NEW)

project(vortex)

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

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic -Wno-dollar-in-identifier-extension")

option(VORTEX_ENABLE_TESTING "Enable building test binary for vortex-cxx" OFF)
option(VORTEX_ENABLE_ASAN "Enable address sanitizer" OFF)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

set(RUST_SOURCE_FILE lib.rs)

# Import Rust crate using Corrosion
corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    FEATURES ${CORROSION_FEATURES}
    CRATES vortex-cxx
)

# Enable CXX bridge for the Rust crate
corrosion_add_cxxbridge(vortex_cxx_bridge CRATE vortex_cxx FILES ${RUST_SOURCE_FILE})

FetchContent_Declare(
    nanoarrow
    GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
    GIT_TAG a579fbf5d192e85b6249935e117de7d02a6dc4e9 # v0.8.0
)
FetchContent_MakeAvailable(nanoarrow)

file(GLOB_RECURSE CPP_SOURCE_FILE CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/*.cpp"
)

# Public headers
set(CPP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Private headers
set(CPP_PRIVATE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
)

# Create the main library combining C++ and Rust code
add_library(vortex STATIC ${CPP_SOURCE_FILE})
target_include_directories(vortex PUBLIC ${CPP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include)
target_include_directories(vortex PRIVATE
    ${CPP_PRIVATE_INCLUDE_DIRS}
)
target_link_libraries(vortex
    PUBLIC nanoarrow_static
    PRIVATE vortex_cxx_bridge
)

if (VORTEX_ENABLE_ASAN)
    target_compile_options(vortex PRIVATE -fsanitize=leak,address,undefined -fno-omit-frame-pointer -fno-common -O1)
    target_link_options(vortex PRIVATE -fsanitize=leak,address,undefined)
endif()

# Tests
if (VORTEX_ENABLE_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(vortex_cxx_test cpp/tests/basic_test.cpp cpp/tests/test_data_generator.cpp)
    target_include_directories(vortex_cxx_test PUBLIC ${CPP_INCLUDE_DIRS})
    target_include_directories(vortex_cxx_test PRIVATE cpp/tests)
    target_link_libraries(vortex_cxx_test PRIVATE gtest_main vortex nanoarrow_static)
    target_include_directories(vortex_cxx_test PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
    )
    # Platform-specific configuration
    if(APPLE)
        set(APPLE_LINK_FLAGS "-framework CoreFoundation -framework Security")
    endif()
    target_link_libraries(vortex_cxx_test PRIVATE vortex_cxx_bridge ${APPLE_LINK_FLAGS})
    if (VORTEX_ENABLE_ASAN)
        target_compile_options(vortex_cxx_test PRIVATE -fsanitize=leak,address,undefined -fno-omit-frame-pointer -fno-common -O1)
        target_link_options(vortex_cxx_test PRIVATE -fsanitize=leak,address,undefined)
    endif()
    include(GoogleTest)
    gtest_discover_tests(vortex_cxx_test)
endif()
