# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# need this for std::float16_t
set(CMAKE_CXX_STANDARD 23)

FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG        v0.9.7
)
FetchContent_MakeAvailable(magic_enum)

file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(vortex_cxx_test ${TEST_FILES})
target_include_directories(vortex_cxx_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vortex_cxx_test PRIVATE
    vortex_cxx_shared
    Catch2::Catch2WithMain
    magic_enum::magic_enum
    nanoarrow_shared)

catch_discover_tests(vortex_cxx_test)
