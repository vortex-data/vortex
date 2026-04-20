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
# https://github.com/catchorg/Catch2/issues/1833
target_compile_definitions(Catch2 PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)

FetchContent_Declare(
  Nanoarrow
  GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow
  GIT_TAG        apache-arrow-nanoarrow-0.8.0
)
FetchContent_MakeAvailable(Nanoarrow)

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
message(NOTICE "Test files ${TEST_FILES}")
add_executable(vortex_ffi_test ${TEST_FILES})
target_link_libraries(vortex_ffi_test PRIVATE
    vortex_ffi_shared
    Catch2::Catch2WithMain
    nanoarrow_shared)
catch_discover_tests(vortex_ffi_test)
