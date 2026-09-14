# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Source-tree entry point: vortex-ffi builds the Rust archive through Cargo;
# lang/cpp adds the C++ wrapper. Both also support standalone configuration.
#
#   add_subdirectory(path/to/vortex vortex)
#   target_link_libraries(my_c_target PRIVATE Vortex::ffi_static)
#   target_link_libraries(my_cpp_target PRIVATE Vortex::cpp_static)
#
# See lang/cpp/README.md for build options and embedding requirements,
# and vortex-ffi/README.md for C API development.

# CMake 3.25 is required for block() and FetchContent SYSTEM.
cmake_minimum_required(VERSION 3.25)

project(Vortex DESCRIPTION "Vortex C and C++ libraries" LANGUAGES NONE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Directory-scoped developer defaults; an embedded root resets inherited ownership.
set(_vortex_top_level "${PROJECT_IS_TOP_LEVEL}")

# A standalone configure writes compile_commands.json for editor tooling.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ctest only descends into subdirectories from a root that has testing enabled.
enable_testing()

add_subdirectory(vortex-ffi ffi)
add_subdirectory(lang/cpp cpp)
