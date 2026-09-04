# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors


FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG        v0.9.7
)
FetchContent_MakeAvailable(magic_enum)

file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(vortex_cxx_test ${TEST_FILES})
# Required by the std::float16_t coverage.
target_compile_features(vortex_cxx_test PRIVATE cxx_std_23)
target_include_directories(vortex_cxx_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vortex_cxx_test PRIVATE
    Vortex::cpp_static
    Catch2::Catch2WithMain
    magic_enum::magic_enum
    nanoarrow_shared)

catch_discover_tests(vortex_cxx_test)

# Validate the private CMake/Rust integration without invoking Cargo.
add_test(NAME vortex_cmake_rust_toolchain
    COMMAND "${CMAKE_COMMAND}"
        "-DVORTEX_CPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/.."
        "-DVORTEX_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RustToolchainTests.cmake")
