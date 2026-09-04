# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Selects the Rust tools and the native Rust target for the Vortex C++ build.

include_guard(GLOBAL)

# Sets VORTEX_RUST_TARGET, VORTEX_RUSTUP_TOOLCHAIN, and
# VORTEX_APPLE_DEPLOYMENT_TARGET in the caller's scope. The cache entries
# VORTEX_CARGO_EXECUTABLE and VORTEX_RUSTC_EXECUTABLE hold the selected tools.
function(_vortex_resolve_rust_toolchain workspace_root)
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
    set(VORTEX_RUST_TARGET "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(VORTEX_RUSTUP_TOOLCHAIN "$ENV{RUSTUP_TOOLCHAIN}" PARENT_SCOPE)

    # Without an explicit deployment target the cc crate uses the SDK version,
    # which can exceed CMake's link target and makes ld64 warn about every
    # Cargo-built C object; 11.0 is rustc's minimum for aarch64-apple-darwin.
    if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(VORTEX_APPLE_DEPLOYMENT_TARGET "11.0" PARENT_SCOPE)
    else()
        set(VORTEX_APPLE_DEPLOYMENT_TARGET "${CMAKE_OSX_DEPLOYMENT_TARGET}" PARENT_SCOPE)
    endif()
endfunction()
