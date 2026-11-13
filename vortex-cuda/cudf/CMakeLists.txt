# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.30.4)
project(vortex_cudf_smoke LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(cudf 26.06 CONFIG REQUIRED COMPONENTS testing)

add_executable(vortex_cudf_smoke smoke.cu)
set_target_properties(vortex_cudf_smoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(vortex_cudf_smoke PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda>"
)
target_link_libraries(vortex_cudf_smoke PRIVATE
    cudf::cudf cudf::cudftestutil_impl GTest::gtest_main CUDA::cudart
)

enable_testing()
add_test(NAME vortex_cudf_smoke COMMAND vortex_cudf_smoke)
set_tests_properties(vortex_cudf_smoke PROPERTIES TIMEOUT 120)
