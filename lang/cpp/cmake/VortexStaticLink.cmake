# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Native link requirements for the source-built Rust static archive. Rust
# archives leave platform-library references for the final linker, so the CMake
# target must publish the validated manifest explicitly. Export policy for an
# enclosing shared library remains the parent's responsibility.

include_guard(GLOBAL)

# Attach the platform libraries required by the Rust static archive; fail if the
# Rust target has no supported link manifest.
function(_vortex_configure_static_link target rust_target)
    if(rust_target MATCHES "^(x86_64|aarch64)-unknown-linux-gnu$")
        if(NOT TARGET Threads::Threads)
            set(THREADS_PREFER_PTHREAD_FLAG TRUE)
            find_package(Threads REQUIRED)
        endif()

        target_link_libraries("${target}" INTERFACE
            gcc_s
            util
            rt
            Threads::Threads
            m
            ${CMAKE_DL_LIBS}
            c)
    elseif(rust_target MATCHES "^(x86_64|aarch64)-apple-darwin$")
        # The public Vortex target is C++, whose driver adds libc++ and
        # libSystem. Publish only additional framework/library requirements.
        find_library(_vortex_core_foundation CoreFoundation REQUIRED NO_CACHE)
        target_link_libraries("${target}" INTERFACE iconv "${_vortex_core_foundation}")
    else()
        message(FATAL_ERROR
            "Vortex has no validated native static-link manifest for Rust target "
            "${rust_target}")
    endif()
endfunction()
