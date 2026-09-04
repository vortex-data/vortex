# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Selects the Rust tools and verifies that Rust, C, and C++ use a supported
# native ABI. It also resolves the macOS SDK settings needed by Cargo builds.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/Helpers.cmake")

# Normalize supported architecture aliases and target prefixes to
# `x86_64` or `aarch64`; lowercase other values for later validation.
function(_vortex_normalize_arch input output)
    string(TOLOWER "${input}" _arch)
    if(_arch MATCHES "^(x86_64|amd64)(-|$)")
        set(_arch "x86_64")
    elseif(_arch MATCHES "^(aarch64|arm64)(-|$)")
        set(_arch "aarch64")
    endif()
    set(${output} "${_arch}" PARENT_SCOPE)
endfunction()

# Assemble the compiler and target-selection arguments needed by probes.
function(_vortex_compiler_command language output)
    set(_command "${CMAKE_${language}_COMPILER}")
    if(CMAKE_${language}_COMPILER_ARG1)
        # Preserve arguments CMake treats as part of the compiler command.
        separate_arguments(_compiler_arg1 NATIVE_COMMAND "${CMAKE_${language}_COMPILER_ARG1}")
        list(APPEND _command ${_compiler_arg1})
    endif()
    if(CMAKE_${language}_COMPILER_TARGET AND
        CMAKE_${language}_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        # Make Clang report the target explicitly selected by CMake.
        list(APPEND _command "--target=${CMAKE_${language}_COMPILER_TARGET}")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES)
        # Make AppleClang report CMake's selected arm64 architecture.
        list(APPEND _command -arch "${CMAKE_OSX_ARCHITECTURES}")
    endif()
    set(${output} "${_command}" PARENT_SCOPE)
endfunction()

# Require the selected compiler to use the Rust target's architecture and ABI.
function(_vortex_validate_native_compiler language rust_target)
    _vortex_compiler_command("${language}" _compiler_command)
    list(APPEND _compiler_command -dumpmachine)
    execute_process(
        COMMAND ${_compiler_command}
        OUTPUT_VARIABLE _compiler_target
        ERROR_VARIABLE _compiler_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _compiler_result)
    if(NOT _compiler_result EQUAL 0 OR _compiler_target STREQUAL "")
        message(FATAL_ERROR
            "Could not determine the target of the CMake ${language} compiler: "
            "${_compiler_error}")
    endif()

    string(TOLOWER "${_compiler_target}" _compiler_target_lower)
    _vortex_normalize_arch("${rust_target}" _rust_arch)
    _vortex_normalize_arch("${_compiler_target_lower}" _compiler_arch)
    if(NOT "${_compiler_arch}" STREQUAL "${_rust_arch}")
        message(FATAL_ERROR
            "CMake ${language} compiler target ${_compiler_target} does not "
            "match Vortex Rust target ${rust_target}")
    endif()

    if(rust_target MATCHES "-unknown-linux-gnu$")
        # GNU distro triples may omit `gnu`, as in `x86_64-redhat-linux`, while
        # incompatible ABIs add a different suffix such as `musl` or `gnux32`.
        if(NOT _compiler_target_lower MATCHES "-linux(-gnu)?$")
            message(FATAL_ERROR
                "CMake ${language} compiler target ${_compiler_target} does not "
                "use the GNU Linux ABI required by Rust target ${rust_target}")
        endif()
    elseif(rust_target MATCHES "-apple-darwin$" AND
        NOT _compiler_target_lower MATCHES
            "-apple-(darwin|macosx)([0-9]+(\\.[0-9]+)*)?$")
        message(FATAL_ERROR
            "CMake ${language} compiler target ${_compiler_target} does not "
            "use the macOS ABI required by Rust target ${rust_target}")
    endif()
endfunction()

# Pin a host Rust tool, resolving the workspace selection behind a rustup proxy.
function(_vortex_resolve_rust_program workspace_root program_name output)
    # find_program skips its search when this inherited scratch name is set.
    set(_program "_program-NOTFOUND")
    find_program(_program
        NAMES "${program_name}"
        NO_CMAKE_FIND_ROOT_PATH
        NO_CACHE)
    if(NOT _program)
        message(FATAL_ERROR
            "Could not find ${program_name} in CMake's program search path")
    endif()

    # Rustup proxies may be hardlinks. RUSTUP_FORCE_ARG0 is rustup's multicall
    # override, so detection does not depend on a filename or symlink target.
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env RUSTUP_FORCE_ARG0=rustup "${_program}" --version
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _proxy_probe_output
        ERROR_VARIABLE _proxy_probe_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _proxy_probe_result)
    if(NOT _proxy_probe_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to inspect ${program_name} at ${_program} "
            "(${_proxy_probe_result}): ${_proxy_probe_error}\n${_proxy_probe_output}")
    endif()

    set(_resolved_program "${_program}")
    set(_is_rustup_proxy OFF)
    if(_proxy_probe_output MATCHES "^rustup [0-9]+\\.[0-9]+")
        set(_is_rustup_proxy ON)
        # Resolve from the workspace so rustup applies the active environment,
        # directory, and toolchain-file selection before CMake pins the binary.
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env
                RUSTUP_FORCE_ARG0=rustup "${_program}" which "${program_name}"
            WORKING_DIRECTORY "${workspace_root}"
            OUTPUT_VARIABLE _resolved_program
            ERROR_VARIABLE _rustup_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _rustup_result)
        if(NOT _rustup_result EQUAL 0)
            message(FATAL_ERROR
                "rustup proxy ${_program} could not resolve active ${program_name} "
                "for ${workspace_root} (${_rustup_result}): ${_rustup_error}\n"
                "${_resolved_program}")
        endif()
    endif()

    if(NOT EXISTS "${_resolved_program}" OR IS_DIRECTORY "${_resolved_program}")
        message(FATAL_ERROR
            "Resolved ${program_name} executable is not a file: ${_resolved_program}")
    endif()
    get_filename_component(_resolved_program "${_resolved_program}" REALPATH)
    _vortex_reject_semicolon("Resolved ${program_name} executable" "${_resolved_program}")
    if(_is_rustup_proxy)
        message(STATUS
            "Vortex resolved rustup ${program_name} proxy ${_program} to ${_resolved_program}")
    endif()
    set(${output} "${_resolved_program}" PARENT_SCOPE)
endfunction()

# Resolve and validate the concrete Cargo and rustc binaries used by the build.
function(_vortex_find_rust_tools workspace_root)
    set(_minimum_version "1.95.0")
    _vortex_resolve_rust_program("${workspace_root}" cargo _cargo)
    _vortex_resolve_rust_program("${workspace_root}" rustc _rustc)

    execute_process(
        COMMAND "${_cargo}" -vV
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _cargo_verbose
        ERROR_VARIABLE _cargo_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _cargo_result)
    if(NOT _cargo_result EQUAL 0)
        message(FATAL_ERROR "Failed to run Cargo at ${_cargo}: ${_cargo_error}")
    endif()

    execute_process(
        COMMAND "${_rustc}" -vV
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _rustc_verbose
        ERROR_VARIABLE _rustc_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rustc_result)
    if(NOT _rustc_result EQUAL 0)
        message(FATAL_ERROR "Failed to run rustc at ${_rustc}: ${_rustc_error}")
    endif()

    string(REGEX MATCH "^cargo ([0-9]+\\.[0-9]+\\.[0-9]+)" _cargo_version_match "${_cargo_verbose}")
    set(_cargo_semver "${CMAKE_MATCH_1}")
    if(_cargo_semver STREQUAL "")
        message(FATAL_ERROR
            "Could not parse Cargo version output from ${_cargo}:\n${_cargo_verbose}")
    elseif("${_cargo_semver}" VERSION_LESS "${_minimum_version}")
        message(FATAL_ERROR
            "Vortex requires Cargo ${_minimum_version} or newer; found:\n${_cargo_verbose}")
    endif()

    # Cargo only orchestrates the build; the pinned rustc host defines its ABI.
    string(REGEX MATCH "host: ([^\r\n]+)" _rustc_host_match "${_rustc_verbose}")
    set(_rustc_host "${CMAKE_MATCH_1}")
    string(REGEX MATCH "release: ([^\r\n]+)" _release_match "${_rustc_verbose}")
    set(_rustc_release "${CMAKE_MATCH_1}")
    if(_rustc_host STREQUAL "" OR _rustc_release STREQUAL "")
        message(FATAL_ERROR
            "Could not parse rustc host and release from ${_rustc}:\n${_rustc_verbose}")
    endif()

    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _rustc_semver "${_rustc_release}")
    if(_rustc_semver STREQUAL "")
        message(FATAL_ERROR
            "Could not parse rustc release ${_rustc_release} from ${_rustc}:\n${_rustc_verbose}")
    elseif("${_rustc_semver}" VERSION_LESS "${_minimum_version}")
        message(FATAL_ERROR
            "Vortex requires rustc ${_minimum_version} or newer; found ${_rustc_release}")
    endif()

    string(REGEX MATCH "^[^\r\n]+" _cargo_version_line "${_cargo_verbose}")
    message(STATUS "Vortex Cargo: ${_cargo_version_line} (${_cargo})")
    message(STATUS "Vortex rustc: ${_rustc_release}, host ${_rustc_host} (${_rustc})")

    set(VORTEX_RESOLVED_CARGO_EXECUTABLE "${_cargo}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_EXECUTABLE "${_rustc}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_RELEASE "${_rustc_release}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_HOST "${_rustc_host}" PARENT_SCOPE)
endfunction()

# Resolve the macOS SDK and deployment target needed by Cargo-built native code.
function(_vortex_resolve_apple_settings sdkroot_output deployment_output)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(${sdkroot_output} "" PARENT_SCOPE)
        set(${deployment_output} "" PARENT_SCOPE)
        return()
    endif()

    set(_apple_sdkroot "")
    if(CMAKE_OSX_SYSROOT)
        if(IS_ABSOLUTE "${CMAKE_OSX_SYSROOT}")
            if(NOT IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
                message(FATAL_ERROR "CMAKE_OSX_SYSROOT is not a directory: ${CMAKE_OSX_SYSROOT}")
            endif()
            file(REAL_PATH "${CMAKE_OSX_SYSROOT}" _apple_sdkroot)
        else()
            # Avoid an inherited scratch value suppressing the host-tool lookup.
            set(_xcrun "_xcrun-NOTFOUND")
            find_program(_xcrun
                NAMES xcrun
                REQUIRED
                NO_CMAKE_FIND_ROOT_PATH
                NO_CACHE)
            execute_process(
                COMMAND "${_xcrun}" --sdk "${CMAKE_OSX_SYSROOT}" --show-sdk-path
                OUTPUT_VARIABLE _apple_sdkroot
                ERROR_VARIABLE _xcrun_error
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _xcrun_result)
            if(NOT _xcrun_result EQUAL 0 OR NOT IS_DIRECTORY "${_apple_sdkroot}")
                message(FATAL_ERROR
                    "Could not resolve CMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT} "
                    "with xcrun: ${_xcrun_error}")
            endif()
            file(REAL_PATH "${_apple_sdkroot}" _apple_sdkroot)
        endif()

        # A Darwin system name also appears in some Apple device toolchains.
        # Restrict the Rust darwin target to an actual macOS SDK.
        get_filename_component(_apple_sdk_name "${_apple_sdkroot}" NAME)
        if(NOT _apple_sdk_name MATCHES
            "^MacOSX([0-9]+(\\.[0-9]+)*)?\\.sdk$")
            message(FATAL_ERROR
                "CMAKE_OSX_SYSROOT must resolve to a macOS SDK; found "
                "${_apple_sdkroot}")
        endif()

        _vortex_reject_semicolon("Resolved macOS SDK root" "${_apple_sdkroot}")
        message(STATUS "Vortex macOS SDK: ${_apple_sdkroot}")
    endif()

    set(_deployment_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    if(NOT _deployment_target)
        # Use the C++ compiler's effective default rather than inferring one from
        # the host OS or Xcode version.
        _vortex_compiler_command(CXX _compiler_command)
        if(_apple_sdkroot)
            list(APPEND _compiler_command -isysroot "${_apple_sdkroot}")
        endif()
        list(APPEND _compiler_command -dM -E -x c++ /dev/null)
        execute_process(
            COMMAND ${_compiler_command}
            OUTPUT_VARIABLE _compiler_defines
            ERROR_VARIABLE _compiler_error
            RESULT_VARIABLE _compiler_result)
        if(NOT _compiler_result EQUAL 0)
            message(FATAL_ERROR
                "Could not determine the default macOS deployment target: "
                "${_compiler_error}")
        endif()
        string(REGEX MATCH
            "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__[ \t]+([0-9]+)"
            _deployment_match "${_compiler_defines}")
        set(_deployment_code "${CMAKE_MATCH_1}")
        if(_deployment_code STREQUAL "")
            message(FATAL_ERROR
                "The C++ compiler did not report a default macOS deployment "
                "target; set CMAKE_OSX_DEPLOYMENT_TARGET")
        endif()

        # Apple encoded 10.9.0 as 1090; macOS 10.10 and later use MMmmpp.
        if(_deployment_code LESS 10000)
            math(EXPR _deployment_major "${_deployment_code} / 100")
            math(EXPR _deployment_minor "(${_deployment_code} / 10) % 10")
            math(EXPR _deployment_patch "${_deployment_code} % 10")
        else()
            math(EXPR _deployment_major "${_deployment_code} / 10000")
            math(EXPR _deployment_minor "(${_deployment_code} / 100) % 100")
            math(EXPR _deployment_patch "${_deployment_code} % 100")
        endif()
        set(_deployment_target "${_deployment_major}.${_deployment_minor}")
        if(NOT _deployment_patch EQUAL 0)
            string(APPEND _deployment_target ".${_deployment_patch}")
        endif()
    endif()
    _vortex_reject_semicolon("macOS deployment target" "${_deployment_target}")
    message(STATUS "Vortex macOS deployment target: ${_deployment_target}")

    set(${sdkroot_output} "${_apple_sdkroot}" PARENT_SCOPE)
    set(${deployment_output} "${_deployment_target}" PARENT_SCOPE)
endfunction()

# Select a supported native Rust target and verify the Rust, C, and C++ ABIs.
function(_vortex_resolve_native_target output)
    # The target mapping assumes every compiler produces code for the build host.
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR
            "Vortex's CMake integration supports native builds only; "
            "CMAKE_CROSSCOMPILING is true")
    endif()
    # Cargo-built native code cannot inherit CMake's generic compile sysroot.
    # CMAKE_SYSROOT_LINK remains valid because CMake owns the final native link.
    if(NOT "${CMAKE_SYSROOT}" STREQUAL "" OR
        NOT "${CMAKE_SYSROOT_COMPILE}" STREQUAL "")
        message(FATAL_ERROR
            "Vortex does not support CMAKE_SYSROOT or CMAKE_SYSROOT_COMPILE; "
            "generic compile sysroots cannot be applied consistently to "
            "Cargo-built code. On macOS, use CMAKE_OSX_SYSROOT instead")
    endif()

    # This integration builds one native Cargo archive and supports only arm64
    # on macOS, so reject x86_64 and multi-architecture lists.
    set(_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES)
        if(NOT "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64")
            message(FATAL_ERROR
                "CMAKE_OSX_ARCHITECTURES must be exactly arm64; CMake selected "
                "${CMAKE_OSX_ARCHITECTURES}")
        endif()
        set(_processor "arm64")
    endif()
    _vortex_normalize_arch("${_processor}" _architecture)

    # Every accepted target has an allowlisted native static-link manifest.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(_architecture STREQUAL "x86_64")
            set(_rust_target "x86_64-unknown-linux-gnu")
        elseif(_architecture STREQUAL "aarch64")
            set(_rust_target "aarch64-unknown-linux-gnu")
        else()
            message(FATAL_ERROR
                "Vortex supports native Linux x86_64 and aarch64 only; "
                "CMake selected ${_processor}")
        endif()

    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        if(NOT _architecture STREQUAL "aarch64")
            message(FATAL_ERROR
                "Vortex macOS development supports arm64 only; "
                "CMake selected ${_processor}")
        endif()
        set(_rust_target "aarch64-apple-darwin")
        message(STATUS
            "Vortex macOS support is development-only and is not a supported "
            "cuDF integration target")
    else()
        message(FATAL_ERROR
            "Vortex's CMake integration supports native Linux and development-only "
            "macOS; CMake selected ${CMAKE_SYSTEM_NAME}")
    endif()

    if(NOT "${VORTEX_RESOLVED_RUSTC_HOST}" STREQUAL "${_rust_target}")
        message(FATAL_ERROR
            "Vortex supports native Rust builds only: rustc host is "
            "${VORTEX_RESOLVED_RUSTC_HOST}, but CMake requires "
            "${_rust_target}")
    endif()

    _vortex_validate_native_compiler(C "${_rust_target}")
    _vortex_validate_native_compiler(CXX "${_rust_target}")
    set(${output} "${_rust_target}" PARENT_SCOPE)
endfunction()
