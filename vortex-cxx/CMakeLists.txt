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

# Platform-specific configuration
# TODO(xinyu): test on mac
if(APPLE)
    target_link_libraries(vortex_cxx "-framework CoreFoundation -framework Security")
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

    # Arrow dependencies
    include(ExternalProject)
    # Define Arrow installation directory
    set(ARROW_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix)
    ExternalProject_Add(
        arrow_ep
        GIT_REPOSITORY https://github.com/apache/arrow.git
        GIT_TAG apache-arrow-20.0.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR cpp
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DARROW_BUILD_SHARED=OFF
            -DARROW_BUILD_STATIC=ON
            -DARROW_DEPENDENCY_SOURCE=AUTO
            -DARROW_MIMALLOC=OFF
            # We'd like a minimal build of Arrow. The following are default options in arrow 20.0.0
            # -DARROW_BUILD_TESTS=OFF
            # -DARROW_BUILD_BENCHMARKS=OFF
            # -DARROW_BUILD_EXAMPLES=OFF
            # -DARROW_BUILD_UTILITIES=OFF
            # -DARROW_COMPUTE=OFF
            # -DARROW_CSV=OFF
            # -DARROW_DATASET=OFF
            # -DARROW_FILESYSTEM=OFF
            # -DARROW_JSON=OFF
            # -DARROW_PARQUET=OFF
            # -DARROW_WITH_BROTLI=OFF
            # -DARROW_WITH_LZ4=OFF
            # -DARROW_WITH_SNAPPY=OFF
            # -DARROW_WITH_ZLIB=OFF
            # -DARROW_WITH_ZSTD=OFF
            # -DARROW_JEMALLOC=OFF
        UPDATE_COMMAND ""
        BUILD_BYPRODUCTS ${ARROW_INSTALL_DIR}/lib/libarrow.a
    )
    file(MAKE_DIRECTORY ${ARROW_INSTALL_DIR}/include)
    add_library(arrow_static STATIC IMPORTED)
    add_dependencies(arrow_static arrow_ep)
    set_target_properties(arrow_static PROPERTIES
        IMPORTED_LOCATION ${ARROW_INSTALL_DIR}/lib/libarrow.a
        INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INSTALL_DIR}/include
    )

    enable_testing()
    add_executable(vortex_cxx_test cpp/tests/basic_test.cpp)
    target_include_directories(vortex_cxx_test PUBLIC ${CPP_INCLUDE_DIRS})
    target_link_libraries(vortex_cxx_test PRIVATE gtest_main vortex_cxx arrow_static)
    # unit tests link to the bridge for the gen_test_data function only
    target_include_directories(vortex_cxx_test PRIVATE 
        ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include
    )
    target_link_libraries(vortex_cxx_test PRIVATE vortex_cxx_bridge) 

    include(GoogleTest)
    gtest_discover_tests(vortex_cxx_test)
endif()