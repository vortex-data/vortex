# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Native link requirements for the source-built Rust static archive. Rust
# archives leave platform-library references for the final linker, so the CMake
# target must publish the validated manifest explicitly. Export policy for an
# enclosing shared library remains the parent's responsibility.

include_guard(GLOBAL)

# Map a supported Rust target to its native link manifest. The platform and
# error outputs are written in PARENT_SCOPE; unknown targets return an error
# rather than guessing from the host.
function(_vortex_static_link_platform rust_target platform_output error_output)
    if(rust_target STREQUAL "x86_64-unknown-linux-gnu" OR
        rust_target STREQUAL "aarch64-unknown-linux-gnu")
        set(_platform "linux")
    elseif(rust_target STREQUAL "x86_64-apple-darwin" OR
        rust_target STREQUAL "aarch64-apple-darwin")
        set(_platform "macos")
    else()
        set(${platform_output} "" PARENT_SCOPE)
        string(CONCAT _platform_error
            "Vortex has no validated native static-link manifest for Rust target "
            "${rust_target}")
        set(${error_output} "${_platform_error}" PARENT_SCOPE)
        return()
    endif()

    set(${platform_output} "${_platform}" PARENT_SCOPE)
    set(${error_output} "" PARENT_SCOPE)
endfunction()

# Attach the native requirements for `rust_target` to the newly created static
# library `target`. Missing targets, unsupported triples, or unavailable system
# libraries are fatal.
function(_vortex_configure_static_link target rust_target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Expected CMake target does not exist: ${target}")
    endif()

    _vortex_static_link_platform("${rust_target}" _platform _platform_error)
    if(_platform_error)
        message(FATAL_ERROR "${_platform_error}")
    endif()

    if(_platform STREQUAL "linux")
        set(THREADS_PREFER_PTHREAD_FLAG TRUE)
        if(NOT TARGET Threads::Threads)
            find_package(Threads REQUIRED)
        endif()

        set(_native_libraries gcc_s util rt Threads::Threads m)
        if(CMAKE_DL_LIBS)
            list(APPEND _native_libraries "${CMAKE_DL_LIBS}")
        endif()
        list(APPEND _native_libraries c)
        target_link_libraries("${target}" INTERFACE ${_native_libraries})
    else()
        # Apple drivers add libSystem implicitly, and the C++ driver also adds
        # libc++. A C final link still needs libc++, so add it only for C links.
        find_library(_vortex_core_foundation CoreFoundation REQUIRED NO_CACHE)
        target_link_libraries("${target}" INTERFACE
            "$<$<LINK_LANGUAGE:C>:c++>"
            iconv
            "${_vortex_core_foundation}")
    endif()
endfunction()
