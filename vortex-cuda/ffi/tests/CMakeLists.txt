# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.30.4)
project(vortex_cudf_tests LANGUAGES CXX)

# Standalone only; deliberately not included by the FFI/CPU build.
option(VORTEX_CUDF_BUILD_TESTS "Build the optional GPU cuDF string dictionary concatenation tests" OFF)
if(NOT VORTEX_CUDF_BUILD_TESTS)
  return()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest CONFIG QUIET)
if(NOT TARGET GTest::gtest_main)
  message(FATAL_ERROR "vortex_cudf_tests requires GTest")
endif()
find_package(cudf 26.06 CONFIG QUIET OPTIONAL_COMPONENTS testing)
if(NOT TARGET cudf::cudf OR NOT TARGET cudf::cudftestutil_impl)
  message(FATAL_ERROR "vortex_cudf_tests requires libcudf and its testing component")
endif()

add_executable(vortex_cudf_tests concatenate_string_dictionaries_test.cpp header_smoke.cpp)
# cudf_test wrappers and the installed test utility sources require nvcc.
# Keep header_smoke.cpp as CXX to also exercise inclusion from an ordinary C++ TU.
set_source_files_properties(concatenate_string_dictionaries_test.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(vortex_cudf_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_include_directories(vortex_cudf_tests PRIVATE ../cinclude)
target_compile_options(vortex_cudf_tests PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda>"
)
target_link_libraries(vortex_cudf_tests PRIVATE
    cudf::cudf cudf::cudftestutil_impl GTest::gtest_main CUDA::cudart
)

enable_testing()
# Avoid discovery-time GPU execution during configuration or build.
add_test(NAME vortex_cudf_tests COMMAND vortex_cudf_tests)
