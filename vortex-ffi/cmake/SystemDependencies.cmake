# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Defines the system libraries required to link the Rust FFI archive into C++
# targets.

include_guard(GLOBAL)

# Attach the platform libraries required by the Rust static archive; fail if the
# Rust target has no supported link manifest.
function(_vortex_attach_system_dependencies target rust_target)
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
    elseif(rust_target STREQUAL "aarch64-apple-darwin")
        # The archive needs no C++ runtime, so C and C++ consumers alike only
        # need libSystem from their driver plus these extra libraries.
        find_library(_vortex_core_foundation CoreFoundation REQUIRED NO_CACHE)
        target_link_libraries("${target}" INTERFACE iconv "${_vortex_core_foundation}")
    else()
        message(FATAL_ERROR
            "Vortex has no validated native static-link manifest for Rust target "
            "${rust_target}")
    endif()
endfunction()
