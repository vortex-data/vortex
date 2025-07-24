# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.22)

project(vortex)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic -Wno-dollar-in-identifier-extension")

option(VORTEX_ENABLE_TESTING "Enable building test binary for vortex-cxx" OFF)

include(FetchContent)
FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

# Import Rust crate using Corrosion
if (VORTEX_ENABLE_TESTING)
    set(CORROSION_FEATURES gen_test_data)
    # Workaround to conditionally generate bindings of the test function *and* compile the test function: https://github.com/dtolnay/cxx/issues/1325
    set(RUST_SOURCE_FILE lib.rs gen_test_data.rs)
else ()
    set(RUST_SOURCE_FILE lib.rs)
endif ()
corrosion_import_crate(
        MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
        FEATURES ${CORROSION_FEATURES}
        CRATES vortex-cxx
)

# Enable CXX bridge for the Rust crate
corrosion_add_cxxbridge(vortex_cxx_bridge CRATE vortex_cxx FILES ${RUST_SOURCE_FILE})

FetchContent_Declare(nanoarrow
        URL "https://www.apache.org/dyn/closer.lua?action=download&filename=arrow/apache-arrow-nanoarrow-0.7.0/apache-arrow-nanoarrow-0.7.0.tar.gz")
FetchContent_MakeAvailable(nanoarrow)

# C++ source files
set(CPP_SOURCE_FILE
        cpp/src/write_options.cpp
        cpp/src/file.cpp
        cpp/src/scan.cpp
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
target_include_directories(vortex PUBLIC ${CPP_INCLUDE_DIRS})
target_include_directories(vortex PRIVATE
        ${CPP_PRIVATE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
        ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/src
)
target_link_libraries(vortex
        PUBLIC nanoarrow_static
        PRIVATE vortex_cxx_bridge
)

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
    add_executable(vortex_cxx_test cpp/tests/basic_test.cpp)
    target_include_directories(vortex_cxx_test PUBLIC ${CPP_INCLUDE_DIRS})
    target_link_libraries(vortex_cxx_test PRIVATE gtest_main vortex nanoarrow_static)
    # unit tests link to the bridge for the gen_test_data function only
    target_include_directories(vortex_cxx_test PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
    )
    # Platform-specific configuration
    if (APPLE)
        set(APPLE_LINK_FLAGS "-framework CoreFoundation -framework Security")
    endif ()
    target_link_libraries(vortex_cxx_test PRIVATE vortex_cxx_bridge ${APPLE_LINK_FLAGS})
    include(GoogleTest)
    gtest_discover_tests(vortex_cxx_test)
endif ()