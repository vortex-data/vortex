# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Public options for the source-only C++ and Cargo integration. Normal variables
# set by an embedding parent take precedence over the cache defaults below.

include_guard(GLOBAL)

option(VORTEX_BUILD_TESTING "Build Vortex C++ tests" OFF)
option(VORTEX_BUILD_EXAMPLES "Build Vortex C++ examples" OFF)
option(VORTEX_CARGO_OFFLINE "Require Cargo to run in offline mode" OFF)

if(NOT DEFINED VORTEX_SANITIZER)
    set(VORTEX_SANITIZER "" CACHE STRING "Instrument Vortex and its consumers with asan or tsan")
endif()
get_property(_sanitizer_is_cached CACHE VORTEX_SANITIZER PROPERTY TYPE SET)
if(_sanitizer_is_cached)
    set_property(CACHE VORTEX_SANITIZER PROPERTY STRINGS "" asan tsan)
endif()

if(NOT DEFINED VORTEX_CARGO_EXECUTABLE)
    set(VORTEX_CARGO_EXECUTABLE "" CACHE FILEPATH "Path to the Cargo executable")
endif()
if(NOT DEFINED VORTEX_RUSTC_EXECUTABLE)
    set(VORTEX_RUSTC_EXECUTABLE "" CACHE FILEPATH "Path to the rustc executable")
endif()
if(NOT DEFINED VORTEX_CARGO_TARGET_DIR)
    set(VORTEX_CARGO_TARGET_DIR "" CACHE PATH "Root for fingerprinted Cargo target directories")
endif()
if(NOT DEFINED VORTEX_CARGO_JOBS)
    set(VORTEX_CARGO_JOBS "" CACHE STRING "Maximum number of nested Cargo jobs")
endif()
if(NOT DEFINED VORTEX_CARGO_FEATURES)
    set(VORTEX_CARGO_FEATURES "" CACHE STRING "Comma-separated vortex-ffi Cargo features")
endif()
if(NOT DEFINED VORTEX_RUSTFLAGS)
    set(VORTEX_RUSTFLAGS "" CACHE STRING "Shell-style Rust flags for the CMake-owned vortex-ffi build")
endif()
