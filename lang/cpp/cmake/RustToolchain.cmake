# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Selects the Rust tools and the native Rust target for the Vortex C++ build.

include_guard(GLOBAL)

# Sets VORTEX_RUST_TARGET, VORTEX_RUSTC_RELEASE, VORTEX_RUSTUP_TOOLCHAIN,
# VORTEX_APPLE_SDKROOT, and VORTEX_APPLE_DEPLOYMENT_TARGET in the caller's
# scope. The cache entries VORTEX_CARGO_EXECUTABLE and VORTEX_RUSTC_EXECUTABLE
# hold the selected tools.
function(_vortex_resolve_rust_toolchain workspace_root)
    # Cargo builds for the rustc host, so C++ must be built for the same machine.
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "Vortex's CMake integration supports native builds only")
    endif()

    # Rustup proxies honor the workspace toolchain file when run from the
    # workspace. RUSTUP_TOOLCHAIN is captured so Cargo builds keep this selection.
    find_program(VORTEX_CARGO_EXECUTABLE NAMES cargo REQUIRED)
    find_program(VORTEX_RUSTC_EXECUTABLE NAMES rustc REQUIRED)
    execute_process(
        COMMAND "${VORTEX_RUSTC_EXECUTABLE}" -vV
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _rustc_verbose
        COMMAND_ERROR_IS_FATAL ANY)
    string(REGEX MATCH "host: ([^\r\n]+)" _match "${_rustc_verbose}")
    set(_host "${CMAKE_MATCH_1}")
    string(REGEX MATCH "release: ([^\r\n]+)" _match "${_rustc_verbose}")
    set(_release "${CMAKE_MATCH_1}")
    message(STATUS "Vortex rustc: ${_release}, host ${_host} (${VORTEX_RUSTC_EXECUTABLE})")

    # Forward CMake's SDK to Cargo-built native code. Without an explicit
    # deployment target the cc crate uses the SDK version, which can exceed
    # CMake's link target and makes ld64 warn about every Cargo-built C object;
    # 11.0 is rustc's minimum for aarch64-apple-darwin.
    set(_sdkroot "")
    set(_deployment_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    if(APPLE)
        if(IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
            set(_sdkroot "${CMAKE_OSX_SYSROOT}")
        endif()
        if(NOT _deployment_target)
            set(_deployment_target "11.0")
        endif()
    endif()

    set(VORTEX_RUST_TARGET "${_host}" PARENT_SCOPE)
    set(VORTEX_RUSTC_RELEASE "${_release}" PARENT_SCOPE)
    set(VORTEX_RUSTUP_TOOLCHAIN "$ENV{RUSTUP_TOOLCHAIN}" PARENT_SCOPE)
    set(VORTEX_APPLE_SDKROOT "${_sdkroot}" PARENT_SCOPE)
    set(VORTEX_APPLE_DEPLOYMENT_TARGET "${_deployment_target}" PARENT_SCOPE)
endfunction()
