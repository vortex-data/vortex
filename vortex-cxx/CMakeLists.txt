# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.22)

project(vortex-cxx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

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
if(VORTEX_ENABLE_TESTING)
    set(CORROSION_FEATURES gen_test_data)
endif()
corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    FEATURES ${CORROSION_FEATURES}
)

# Enable CXX bridge for the Rust crate
corrosion_add_cxxbridge(vortex_cxx_bridge CRATE vortex_cxx_rs FILES lib.rs)

FetchContent_Declare(nanoarrow
    URL "https://www.apache.org/dyn/closer.lua?action=download&filename=arrow/apache-arrow-nanoarrow-0.7.0/apache-arrow-nanoarrow-0.7.0.tar.gz")
FetchContent_MakeAvailable(nanoarrow)

# C++ source files
set(CPP_SOURCE_FILE
    cpp/write_options.cpp
    cpp/file.cpp
    cpp/runtime.cpp
)

# Public headers
set(CPP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Create the main library combining C++ and Rust code
add_library(vortex_cxx STATIC ${CPP_SOURCE_FILE})
target_include_directories(vortex_cxx PUBLIC ${CPP_INCLUDE_DIRS})
target_include_directories(vortex_cxx PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/src
)
target_link_libraries(vortex_cxx 
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
    target_link_libraries(vortex_cxx_test PRIVATE gtest_main vortex_cxx nanoarrow_static)
    # unit tests link to the bridge for the gen_test_data function only
    target_include_directories(vortex_cxx_test PRIVATE 
        ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
    )
    # Platform-specific configuration
    if(APPLE)
        set(MAC_LINK_FLAGS "-framework CoreFoundation -framework Security")
    endif()
    target_link_libraries(vortex_cxx_test PRIVATE vortex_cxx_bridge ${MAC_LINK_FLAGS})
    include(GoogleTest)
    gtest_discover_tests(vortex_cxx_test)
endif()