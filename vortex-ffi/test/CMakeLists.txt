# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

FetchContent_Declare(
    Catch
    SYSTEM
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1)
FetchContent_MakeAvailable(Catch)
# By path: Catch2 exports its module directory only to the directory that
# populated it, which may be a parent that fetched Catch2 first.
include("${catch_SOURCE_DIR}/extras/Catch.cmake")

# Catch2 enables std::string_view, optional, variant, and byte support from
# C++17, in headers and compiled sources alike; build it with those on so it
# matches the C++20 tests that include its headers.
target_compile_features(Catch2 PRIVATE cxx_std_17)
# https://github.com/catchorg/Catch2/issues/1833
target_compile_definitions(Catch2 PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)

file(GLOB _vortex_ffi_test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(vortex_ffi_test ${_vortex_ffi_test_sources})
target_compile_features(vortex_ffi_test PRIVATE cxx_std_20)
target_compile_options(vortex_ffi_test PRIVATE -Wall -Wextra -Wpedantic)
if(VORTEX_WARNINGS_AS_ERRORS)
    target_compile_options(vortex_ffi_test PRIVATE -Werror)
endif()
target_link_libraries(vortex_ffi_test PRIVATE
    Vortex::ffi_static
    Catch2::Catch2WithMain
    nanoarrow_shared)

catch_discover_tests(vortex_ffi_test)
