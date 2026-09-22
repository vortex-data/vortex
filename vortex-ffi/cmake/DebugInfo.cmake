# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

include_guard(DIRECTORY)

set(VORTEX_DEBUG_INFO "2" CACHE STRING
    "Debug information for C, C++, and Rust: 0 (none), 1 (limited), 2 (full)")
set_property(CACHE VORTEX_DEBUG_INFO PROPERTY STRINGS 0 1 2)
if(NOT VORTEX_DEBUG_INFO MATCHES "^[012]$")
    message(FATAL_ERROR "VORTEX_DEBUG_INFO must be 0, 1, or 2")
endif()

# Keep the override in Vortex's directories, including fetched dependencies,
# without changing unrelated targets in an embedding parent.
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-g${VORTEX_DEBUG_INFO}>")
