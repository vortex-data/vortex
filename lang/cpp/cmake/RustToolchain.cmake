# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Resolves and validates the Rust and native toolchains used by the Vortex C++
# build.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/Helpers.cmake")

# Normalize supported architecture aliases and target prefixes to `x86_64` or
# `aarch64`; lowercase other values for later validation.
function(_vortex_normalize_arch input output)
    string(TOLOWER "${input}" _arch)
    if(_arch MATCHES "^(x86_64|amd64)(-|$)")
        set(_arch "x86_64")
    elseif(_arch MATCHES "^(aarch64|arm64)(-|$)")
        set(_arch "aarch64")
    endif()
    set(${output} "${_arch}" PARENT_SCOPE)
endfunction()

# Assemble a compiler probe command, including CMAKE_<LANG>_COMPILER_ARG1 and
# Clang's explicit CMAKE_<LANG>_COMPILER_TARGET.
function(_vortex_compiler_command language output)
    set(_compiler "${CMAKE_${language}_COMPILER}")
    set(_command "${_compiler}")
    if(CMAKE_${language}_COMPILER_ARG1)
        separate_arguments(_compiler_arg1 NATIVE_COMMAND "${CMAKE_${language}_COMPILER_ARG1}")
        list(APPEND _command ${_compiler_arg1})
    endif()
    if(CMAKE_${language}_COMPILER_TARGET AND CMAKE_${language}_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        list(APPEND _command "--target=${CMAKE_${language}_COMPILER_TARGET}")
    endif()
    set(${output} "${_command}" PARENT_SCOPE)
endfunction()

# Probe the selected compiler with `-dumpmachine` and require its architecture
# and platform family to match the Rust target.
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

    _vortex_normalize_arch("${rust_target}" _rust_arch)
    _vortex_normalize_arch("${_compiler_target}" _compiler_arch)
    if(NOT _compiler_arch STREQUAL _rust_arch)
        message(FATAL_ERROR
            "CMake ${language} compiler target ${_compiler_target} does not "
            "match Vortex Rust target ${rust_target}")
    endif()
    if(rust_target MATCHES "-unknown-linux-gnu$" AND
        (NOT _compiler_target MATCHES "linux" OR _compiler_target MATCHES "musl"))
        message(FATAL_ERROR
            "CMake ${language} compiler target ${_compiler_target} is not "
            "compatible with GNU Rust target ${rust_target}")
    elseif(rust_target MATCHES "-apple-darwin$" AND NOT _compiler_target MATCHES "(apple|darwin)")
        message(FATAL_ERROR
            "CMake ${language} compiler target ${_compiler_target} is not "
            "compatible with Rust target ${rust_target}")
    endif()
endfunction()

# Find a Rust tool with find_program() and return its absolute path.
function(_vortex_find_rust_program output program_name)
    find_program(_program NAMES "${program_name}" NO_CACHE)
    if(NOT _program)
        message(FATAL_ERROR "Could not find ${program_name} in PATH")
    endif()
    get_filename_component(_program "${_program}" ABSOLUTE)
    _vortex_reject_semicolon("Resolved ${program_name} executable" "${_program}")
    set(${output} "${_program}" PARENT_SCOPE)
endfunction()

# Resolve a Cargo or rustc candidate to the concrete executable selected for the
# workspace.
function(_vortex_resolve_rustup_proxy program program_name workspace_root output)
    # A rustup proxy normally dispatches from argv[0]. Forcing that identity to
    # `rustup` makes the proxy identify its manager, while a real tool still
    # reports itself, so detection does not depend on the candidate's filename.
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env RUSTUP_FORCE_ARG0=rustup "${program}" --version
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _proxy_probe_output
        ERROR_VARIABLE _proxy_probe_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _proxy_probe_result)
    if(NOT _proxy_probe_result EQUAL 0)
        message(FATAL_ERROR "Failed to inspect ${program_name} at ${program}: ${_proxy_probe_error}")
    endif()

    if(NOT _proxy_probe_output MATCHES "^rustup [0-9]+\\.[0-9]+")
        get_filename_component(_resolved_program "${program}" REALPATH)
        set(${output} "${_resolved_program}" PARENT_SCOPE)
        return()
    endif()

    get_filename_component(_proxy_directory "${program}" DIRECTORY)
    find_program(_rustup
        NAMES rustup
        HINTS "${_proxy_directory}"
        NO_DEFAULT_PATH
        NO_CACHE)
    if(NOT _rustup)
        message(FATAL_ERROR "${program} is a rustup proxy, but rustup was not found next to it")
    endif()
    # rustup selection honors directory overrides and rust-toolchain files. Query
    # from the workspace so the resolved binary matches the later Cargo
    # invocation.
    execute_process(
        COMMAND "${_rustup}" which "${program_name}"
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _resolved_program
        ERROR_VARIABLE _rustup_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rustup_result)
    if(NOT _rustup_result EQUAL 0 OR
        NOT EXISTS "${_resolved_program}" OR
        IS_DIRECTORY "${_resolved_program}")
        message(FATAL_ERROR
            "rustup could not resolve the active ${program_name} toolchain "
            "for ${workspace_root}: ${_rustup_error}")
    endif()
    get_filename_component(_resolved_program "${_resolved_program}" REALPATH)
    _vortex_reject_semicolon("Resolved rustup ${program_name} executable" "${_resolved_program}")
    message(STATUS "Vortex resolved rustup ${program_name} proxy ${program} to ${_resolved_program}")
    set(${output} "${_resolved_program}" PARENT_SCOPE)
endfunction()

# Resolve Cargo and rustc for the workspace, require both to be at least version
# 1.95.0 with matching host triples, and return the tools, rustc release, and
# host triple to the caller.
function(_vortex_find_rust_tools workspace_root)
    _vortex_find_rust_program(_cargo_candidate cargo)
    _vortex_find_rust_program(_rustc_candidate rustc)

    _vortex_resolve_rustup_proxy("${_cargo_candidate}" cargo "${workspace_root}" _cargo)
    _vortex_resolve_rustup_proxy("${_rustc_candidate}" rustc "${workspace_root}" _rustc)

    # Query both concrete binaries: minimum versions establish required behavior,
    # while matching host triples reject mixed non-rustup installations as well.
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
    if(_cargo_semver STREQUAL "" OR _cargo_semver VERSION_LESS "1.95.0")
        message(FATAL_ERROR "Vortex requires Cargo 1.95.0 or newer; found:\n${_cargo_verbose}")
    endif()

    string(REGEX MATCH "host: ([^\r\n]+)" _cargo_host_match "${_cargo_verbose}")
    set(_cargo_host "${CMAKE_MATCH_1}")
    string(REGEX MATCH "host: ([^\r\n]+)" _rustc_host_match "${_rustc_verbose}")
    set(_rustc_host "${CMAKE_MATCH_1}")
    string(REGEX MATCH "release: ([^\r\n]+)" _release_match "${_rustc_verbose}")
    set(_rustc_release "${CMAKE_MATCH_1}")
    if(_cargo_host STREQUAL "" OR _rustc_host STREQUAL "" OR _rustc_release STREQUAL "")
        message(FATAL_ERROR "Could not parse Cargo/rustc verbose version output from ${_cargo} and ${_rustc}")
    endif()
    if(NOT _cargo_host STREQUAL _rustc_host)
        message(FATAL_ERROR
            "Cargo host ${_cargo_host} does not match rustc host "
            "${_rustc_host}; ensure both resolve from the same toolchain")
    endif()
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _rustc_semver "${_rustc_release}")
    if(_rustc_semver VERSION_LESS "1.95.0")
        message(FATAL_ERROR "Vortex requires rustc 1.95.0 or newer; found ${_rustc_release}")
    endif()

    string(REGEX MATCH "^[^\r\n]+" _cargo_version_line "${_cargo_verbose}")
    message(STATUS "Vortex Cargo: ${_cargo_version_line} (${_cargo})")
    message(STATUS "Vortex rustc: ${_rustc_release}, host ${_rustc_host} (${_rustc})")

    set(VORTEX_RESOLVED_CARGO_EXECUTABLE "${_cargo}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_EXECUTABLE "${_rustc}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_RELEASE "${_rustc_release}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_HOST "${_rustc_host}" PARENT_SCOPE)
endfunction()

# Resolve configured Apple SDK details and the effective macOS deployment target;
# return empty values on non-Apple platforms.
function(_vortex_resolve_apple_settings root_output deployment_output)
    if(NOT APPLE)
        set(${root_output} "" PARENT_SCOPE)
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
            find_program(_xcrun NAMES xcrun REQUIRED NO_CACHE)
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

        _vortex_reject_semicolon("Resolved Apple SDK root" "${_apple_sdkroot}")
        message(STATUS "Vortex Apple SDK: ${_apple_sdkroot}")
    endif()

    set(_deployment_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    if(NOT _deployment_target)
        # Match Rust and Cargo-built native code to the compiler's effective
        # default instead of guessing from the host OS or Xcode version.
        _vortex_compiler_command(C _compiler_command)
        if(_apple_sdkroot)
            list(APPEND _compiler_command -isysroot "${_apple_sdkroot}")
        endif()
        list(APPEND _compiler_command -dM -E -x c /dev/null)
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
                "The C compiler did not report a default macOS deployment "
                "target; set CMAKE_OSX_DEPLOYMENT_TARGET")
        endif()
        math(EXPR _deployment_major "${_deployment_code} / 10000")
        math(EXPR _deployment_minor "(${_deployment_code} / 100) % 100")
        math(EXPR _deployment_patch "${_deployment_code} % 100")
        set(_deployment_target "${_deployment_major}.${_deployment_minor}")
        if(NOT _deployment_patch EQUAL 0)
            string(APPEND _deployment_target ".${_deployment_patch}")
        endif()
    endif()
    _vortex_reject_semicolon("macOS deployment target" "${_deployment_target}")
    message(STATUS "Vortex macOS deployment target: ${_deployment_target}")

    set(${root_output} "${_apple_sdkroot}" PARENT_SCOPE)
    set(${deployment_output} "${_deployment_target}" PARENT_SCOPE)
endfunction()

# Select a supported native Rust target for CMake's platform and architecture,
# then validate it against the rustc, C, and C++ targets.
function(_vortex_resolve_native_target output)
    # Native C dependencies built under Cargo must share the CMake host ABI.
    # Reject cross builds explicitly rather than implying support from a plausible
    # triple.
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR
            "Vortex's initial CMake integration supports native builds only; "
            "CMAKE_CROSSCOMPILING is true")
    endif()

    # Apple can override the generic processor selection, but one Rust static
    # archive cannot represent a multi-architecture universal binary.
    set(_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        list(LENGTH CMAKE_OSX_ARCHITECTURES _architecture_count)
        if(NOT _architecture_count EQUAL 1)
            message(FATAL_ERROR
                "Vortex does not support Apple universal binaries; set "
                "exactly one CMAKE_OSX_ARCHITECTURES value")
        endif()
        list(GET CMAKE_OSX_ARCHITECTURES 0 _processor)
    endif()
    _vortex_normalize_arch("${_processor}" _architecture)

    # Keep this mapping closed: every accepted triple has a separately allowlisted
    # native static-link manifest.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(_architecture STREQUAL "x86_64")
            set(_rust_target "x86_64-unknown-linux-gnu")
        elseif(_architecture STREQUAL "aarch64")
            set(_rust_target "aarch64-unknown-linux-gnu")
        else()
            message(FATAL_ERROR
                "Vortex supports native Linux x86_64 and aarch64 only; CMake selected ${_processor}")
        endif()

    elseif(APPLE)
        if(_architecture STREQUAL "x86_64")
            set(_rust_target "x86_64-apple-darwin")
        elseif(_architecture STREQUAL "aarch64")
            set(_rust_target "aarch64-apple-darwin")
        else()
            message(FATAL_ERROR
                "Vortex standalone macOS builds support x86_64 and arm64 only; "
                "CMake selected ${_processor}")
        endif()
        message(STATUS
            "Vortex macOS support is for standalone development only; "
            "it is not a supported cuDF integration target")
    else()
        message(FATAL_ERROR
            "Vortex's initial CMake integration supports native Linux and "
            "standalone macOS; CMake selected ${CMAKE_SYSTEM_NAME}")
    endif()

    if(NOT VORTEX_RESOLVED_RUSTC_HOST STREQUAL _rust_target)
        message(FATAL_ERROR
            "Vortex supports native Rust builds only: rustc host is "
            "${VORTEX_RESOLVED_RUSTC_HOST}, but CMake requires "
            "${_rust_target}")
    endif()

    _vortex_validate_native_compiler(C "${_rust_target}")
    _vortex_validate_native_compiler(CXX "${_rust_target}")
    set(${output} "${_rust_target}" PARENT_SCOPE)
endfunction()
