# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
include(CTest)
include(FetchContent)

FetchContent_Declare(
  Catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch)
include(Catch)

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
message(NOTICE "Test files ${TEST_FILES}")
add_executable(vortex_ffi_test ${TEST_FILES})
target_link_libraries(vortex_ffi_test PRIVATE vortex_ffi_shared Catch2::Catch2WithMain)
catch_discover_tests(vortex_ffi_test)
