# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Rust toolchain and native-ABI validation for Cargo-backed Vortex CMake builds.
# This module resolves concrete tools and one supported native Rust target
# before build rules are created; it neither installs toolchains nor enables
# cross builds.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/VortexHelpers.cmake")

# Normalize a processor or target-like value to Vortex's architecture spelling.
# Unknown values pass through lowercased so callers can report the original
# unsupported selection.
function(_vortex_normalize_arch input output)
    string(TOLOWER "${input}" _arch)
    if(_arch MATCHES "^(x86_64|amd64)(-|$)")
        set(_arch "x86_64")
    elseif(_arch MATCHES "^(aarch64|arm64)(-|$)")
        set(_arch "aarch64")
    endif()
    set(${output} "${_arch}" PARENT_SCOPE)
endfunction()

# Resolve a Cargo-family executable candidate. `configured_value` is an absolute
# path or command name; an empty value selects `program_name`. Names search
# normal system paths before conventional Cargo homes. The variable named by
# `output` receives an absolute path in PARENT_SCOPE; a missing or non-file
# result is fatal.
function(_vortex_find_rust_program output program_name configured_value)
    unset(_program)
    if(configured_value)
        if(IS_ABSOLUTE "${configured_value}")
            set(_program "${configured_value}")
        else()
            find_program(_program NAMES "${configured_value}" NO_CACHE)
        endif()
    else()
        find_program(_program NAMES "${program_name}" NO_CACHE)
    endif()

    # Explicit paths and normal command lookup take precedence. Cargo homes are a
    # fallback for embedding environments whose PATH omits rustup's installation.
    if(NOT _program)
        set(_rust_program_hints)
        if(DEFINED ENV{CARGO_HOME} AND NOT "$ENV{CARGO_HOME}" STREQUAL "")
            _vortex_reject_semicolon("CARGO_HOME" "$ENV{CARGO_HOME}")
            list(APPEND _rust_program_hints "$ENV{CARGO_HOME}/bin")
        endif()
        if(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
            _vortex_reject_semicolon("HOME" "$ENV{HOME}")
            list(APPEND _rust_program_hints "$ENV{HOME}/.cargo/bin")
        endif()
        if(configured_value)
            set(_program_names "${configured_value}")
        else()
            set(_program_names "${program_name}")
        endif()
        find_program(_program
            NAMES ${_program_names}
            HINTS ${_rust_program_hints}
            NO_DEFAULT_PATH
            NO_CACHE)
    endif()

    if(NOT _program OR NOT EXISTS "${_program}" OR IS_DIRECTORY "${_program}")
        if(configured_value)
            message(FATAL_ERROR "Configured ${program_name} executable does not exist: ${configured_value}")
        else()
            message(FATAL_ERROR "Could not find ${program_name} in PATH, CARGO_HOME/bin, or HOME/.cargo/bin")
        endif()
    endif()
    get_filename_component(_program "${_program}" ABSOLUTE)
    _vortex_reject_semicolon("Resolved ${program_name} executable" "${_program}")
    set(${output} "${_program}" PARENT_SCOPE)
endfunction()

# Turn `program` into the concrete `program_name` binary selected for
# `workspace_root`. The variables named by `output` and `proxy_output` receive
# the real path and a proxy boolean in PARENT_SCOPE. Probes or failed resolution
# are fatal; there is no recoverable result.
function(_vortex_resolve_rustup_proxy program program_name workspace_root output proxy_output)
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
        set(${proxy_output} FALSE PARENT_SCOPE)
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
    set(${proxy_output} TRUE PARENT_SCOPE)
endfunction()

# Resolve and validate the Cargo/rustc pair used for `workspace_root`. Optional
# inputs are VORTEX_CARGO_EXECUTABLE and VORTEX_RUSTC_EXECUTABLE; defaults are
# cached as user-visible FILEPATH values. VORTEX_RESOLVED_CARGO_EXECUTABLE,
# VORTEX_RESOLVED_CARGO_VERBOSE, VORTEX_RESOLVED_RUSTC_EXECUTABLE,
# VORTEX_RESOLVED_RUSTC_VERBOSE, VORTEX_RESOLVED_RUSTC_RELEASE, and
# VORTEX_RESOLVED_RUSTC_HOST are written to PARENT_SCOPE. Probes emit status;
# execution, parse, minimum-version, or host-coherence failures are fatal.
function(_vortex_find_rust_tools workspace_root)
    _vortex_reject_semicolon("VORTEX_CARGO_EXECUTABLE" "${VORTEX_CARGO_EXECUTABLE}")
    _vortex_reject_semicolon("VORTEX_RUSTC_EXECUTABLE" "${VORTEX_RUSTC_EXECUTABLE}")

    _vortex_find_rust_program(_cargo_candidate cargo "${VORTEX_CARGO_EXECUTABLE}")
    _vortex_find_rust_program(_rustc_candidate rustc "${VORTEX_RUSTC_EXECUTABLE}")
    if(NOT VORTEX_CARGO_EXECUTABLE)
        set(VORTEX_CARGO_EXECUTABLE "${_cargo_candidate}" CACHE FILEPATH
            "Path to the Cargo executable" FORCE)
    endif()
    if(NOT VORTEX_RUSTC_EXECUTABLE)
        set(VORTEX_RUSTC_EXECUTABLE "${_rustc_candidate}" CACHE FILEPATH
            "Path to the rustc executable" FORCE)
    endif()

    _vortex_resolve_rustup_proxy("${_cargo_candidate}" cargo "${workspace_root}" _cargo _cargo_was_proxy)
    _vortex_resolve_rustup_proxy("${_rustc_candidate}" rustc "${workspace_root}" _rustc _rustc_was_proxy)
    # Two rustup-managed tools must come from one bin directory; otherwise Cargo
    # could silently invoke a compiler from a different selected toolchain.
    if(_cargo_was_proxy AND _rustc_was_proxy)
        get_filename_component(_cargo_bin_directory "${_cargo}" DIRECTORY)
        get_filename_component(_rustc_bin_directory "${_rustc}" DIRECTORY)
        if(NOT _cargo_bin_directory STREQUAL _rustc_bin_directory)
            message(FATAL_ERROR
                "rustup resolved Cargo and rustc from different toolchains: "
                "${_cargo_bin_directory} and ${_rustc_bin_directory}")
        endif()
    endif()

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
            "${_rustc_host}; "
            "select a compatible VORTEX_CARGO_EXECUTABLE and "
            "VORTEX_RUSTC_EXECUTABLE pair")
    endif()
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _rustc_semver "${_rustc_release}")
    if(_rustc_semver VERSION_LESS "1.95.0")
        message(FATAL_ERROR "Vortex requires rustc 1.95.0 or newer; found ${_rustc_release}")
    endif()

    string(REGEX MATCH "^[^\r\n]+" _cargo_version_line "${_cargo_verbose}")
    message(STATUS "Vortex Cargo: ${_cargo_version_line} (${_cargo})")
    message(STATUS "Vortex rustc: ${_rustc_release}, host ${_rustc_host} (${_rustc})")

    set(VORTEX_RESOLVED_CARGO_EXECUTABLE "${_cargo}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_CARGO_VERBOSE "${_cargo_verbose}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_EXECUTABLE "${_rustc}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_VERBOSE "${_rustc_verbose}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_RELEASE "${_rustc_release}" PARENT_SCOPE)
    set(VORTEX_RESOLVED_RUSTC_HOST "${_rustc_host}" PARENT_SCOPE)
endfunction()

# Determine the effective Apple deployment target. The variable named by
# `output` receives the explicit CMake value, a compiler-derived default, or an
# empty string off Apple in PARENT_SCOPE. Compiler execution/parsing failures
# are fatal and status messages are emitted for the selected value.
function(_vortex_resolve_apple_deployment_target output)
    if(NOT APPLE)
        set(${output} "" PARENT_SCOPE)
        return()
    endif()

    if(CMAKE_OSX_DEPLOYMENT_TARGET)
        message(STATUS "Vortex macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
        set(${output} "${CMAKE_OSX_DEPLOYMENT_TARGET}" PARENT_SCOPE)
        return()
    endif()

    # Probe the compiler's predefined minimum-version macro with CMake's effective
    # target and sysroot. This captures the driver's real default instead of
    # guessing it from the host OS or Xcode version.
    _vortex_optional_value(CMAKE_C_COMPILER_ARG1 _compiler_arg1_value)
    _vortex_optional_value(CMAKE_C_COMPILER_TARGET _compiler_target)
    set(_compiler_command "${CMAKE_C_COMPILER}")
    if(_compiler_arg1_value)
        separate_arguments(_compiler_arg1 NATIVE_COMMAND "${_compiler_arg1_value}")
        list(APPEND _compiler_command ${_compiler_arg1})
    endif()
    if(_compiler_target AND CMAKE_C_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        list(APPEND _compiler_command "--target=${_compiler_target}")
    endif()
    if(CMAKE_OSX_SYSROOT)
        list(APPEND _compiler_command "-isysroot" "${CMAKE_OSX_SYSROOT}")
    endif()
    list(APPEND _compiler_command -dM -E -x c /dev/null)
    execute_process(
        COMMAND ${_compiler_command}
        OUTPUT_VARIABLE _compiler_defines
        ERROR_VARIABLE _compiler_defines_error
        RESULT_VARIABLE _compiler_defines_result)
    if(NOT _compiler_defines_result EQUAL 0)
        message(FATAL_ERROR
            "Could not determine Apple deployment target from "
            "${CMAKE_C_COMPILER}: ${_compiler_defines_error}. "
            "Set CMAKE_OSX_DEPLOYMENT_TARGET explicitly.")
    endif()

    string(REGEX MATCH
        "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__[ \t]+([0-9]+)"
        _deployment_match "${_compiler_defines}")
    set(_deployment_code "${CMAKE_MATCH_1}")
    if(_deployment_code STREQUAL "")
        message(FATAL_ERROR
            "The Apple compiler did not report its default macOS "
            "deployment target. Set CMAKE_OSX_DEPLOYMENT_TARGET explicitly.")
    endif()

    math(EXPR _deployment_major "${_deployment_code} / 10000")
    math(EXPR _deployment_minor "(${_deployment_code} / 100) % 100")
    math(EXPR _deployment_patch "${_deployment_code} % 100")
    if(_deployment_patch EQUAL 0)
        set(_deployment_target "${_deployment_major}.${_deployment_minor}")
    else()
        set(_deployment_target "${_deployment_major}.${_deployment_minor}.${_deployment_patch}")
    endif()
    message(STATUS "Vortex macOS deployment target: ${_deployment_target}")
    set(${output} "${_deployment_target}" PARENT_SCOPE)
endfunction()

# Resolve CMAKE_OSX_SYSROOT and its SDK version. The variables named by
# `root_output` and `version_output` receive values in PARENT_SCOPE, or empty
# strings off Apple/without a sysroot. xcrun lookup or invalid SDK metadata is
# fatal; successful resolution emits a status message.
function(_vortex_resolve_apple_sdkroot root_output version_output)
    if(NOT APPLE OR NOT CMAKE_OSX_SYSROOT)
        set(${root_output} "" PARENT_SCOPE)
        set(${version_output} "" PARENT_SCOPE)
        return()
    endif()

    # CMake accepts either a concrete SDK directory or a name such as `macosx`.
    # Preserve and canonicalize an absolute selection; let xcrun resolve a named
    # SDK. xcrun remains authoritative for the selected SDK's version in both
    # cases.
    find_program(_xcrun NAMES xcrun REQUIRED NO_CACHE)
    if(IS_ABSOLUTE "${CMAKE_OSX_SYSROOT}")
        if(NOT IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
            message(FATAL_ERROR "CMAKE_OSX_SYSROOT is not a directory: ${CMAKE_OSX_SYSROOT}")
        endif()
        file(REAL_PATH "${CMAKE_OSX_SYSROOT}" _apple_sdkroot)
    else()
        execute_process(
            COMMAND "${_xcrun}" --sdk "${CMAKE_OSX_SYSROOT}" --show-sdk-path
            OUTPUT_VARIABLE _apple_sdkroot
            ERROR_VARIABLE _xcrun_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _xcrun_result)
        if(NOT _xcrun_result EQUAL 0 OR NOT IS_DIRECTORY "${_apple_sdkroot}")
            message(FATAL_ERROR "Could not resolve CMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT} with xcrun: ${_xcrun_error}")
        endif()
        file(REAL_PATH "${_apple_sdkroot}" _apple_sdkroot)
    endif()

    execute_process(
        COMMAND "${_xcrun}" --sdk "${CMAKE_OSX_SYSROOT}" --show-sdk-version
        OUTPUT_VARIABLE _apple_sdk_version
        ERROR_VARIABLE _xcrun_version_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _xcrun_version_result)
    if(NOT _xcrun_version_result EQUAL 0 OR _apple_sdk_version STREQUAL "")
        message(FATAL_ERROR
            "Could not determine the version of Apple SDK "
            "${CMAKE_OSX_SYSROOT}: ${_xcrun_version_error}")
    endif()

    _vortex_reject_semicolon("Resolved Apple SDK root" "${_apple_sdkroot}")
    _vortex_reject_semicolon("Resolved Apple SDK version" "${_apple_sdk_version}")
    message(STATUS "Vortex Apple SDK: ${_apple_sdk_version} (${_apple_sdkroot})")
    set(${root_output} "${_apple_sdkroot}" PARENT_SCOPE)
    set(${version_output} "${_apple_sdk_version}" PARENT_SCOPE)
endfunction()

# Validate one native compiler driver against `architecture` and `rust_target`.
# `compiler`, optional `arg1`/`configured_target`, `compiler_id`, and diagnostic
# `label` describe the CMake compiler invocation. There are no outputs; an
# unusable driver or architecture/platform/environment mismatch is fatal.
function(_vortex_validate_native_compiler
    compiler arg1 configured_target compiler_id architecture rust_target label)
    set(_compiler_command "${compiler}")
    if(arg1)
        separate_arguments(_compiler_arg1 NATIVE_COMMAND "${arg1}")
        list(APPEND _compiler_command ${_compiler_arg1})
    endif()
    if(configured_target AND compiler_id MATCHES "^(AppleClang|Clang)$")
        list(APPEND _compiler_command "--target=${configured_target}")
    endif()
    # `-dumpmachine` checks the driver's effective default tuple, which compiler
    # identity and CMAKE_*_COMPILER_TARGET declarations alone do not guarantee.
    list(APPEND _compiler_command -dumpmachine)
    execute_process(
        COMMAND ${_compiler_command}
        OUTPUT_VARIABLE _compiler_target
        ERROR_VARIABLE _compiler_target_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _compiler_target_result)
    if(NOT _compiler_target_result EQUAL 0 OR _compiler_target STREQUAL "")
        message(FATAL_ERROR
            "Could not determine the target of ${label} compiler ${compiler}: "
            "${_compiler_target_error}")
    endif()

    _vortex_normalize_arch("${_compiler_target}" _compiler_arch)
    if(NOT _compiler_arch STREQUAL architecture)
        message(FATAL_ERROR
            "${label} compiler target ${_compiler_target} does not match "
            "Vortex Rust target ${rust_target}")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
        (NOT _compiler_target MATCHES "linux" OR
        _compiler_target MATCHES "musl"))
        message(FATAL_ERROR
            "${label} compiler target ${_compiler_target} is not compatible "
            "with GNU Rust target ${rust_target}")
    elseif(APPLE AND NOT _compiler_target MATCHES "(apple|darwin)")
        message(FATAL_ERROR
            "${label} compiler target ${_compiler_target} is not compatible "
            "with Rust target ${rust_target}")
    endif()
    if(configured_target)
        _vortex_normalize_arch("${configured_target}" _configured_arch)
        if(NOT _configured_arch STREQUAL architecture)
            message(FATAL_ERROR
                "Configured ${label} compiler target ${configured_target} does "
                "not match Vortex Rust target ${rust_target}")
        endif()
    endif()
endfunction()

# Select and validate one supported native Rust target. `rustc` and
# `workspace_root` drive Rust probes; CMake platform/compiler state and
# VORTEX_RESOLVED_RUSTC_HOST constrain the result. The variable named by
# `output` receives the triple in PARENT_SCOPE. Cross, universal, unsupported,
# missing-std, compiler-probe, and ABI disagreements are fatal; probes may emit
# status output.
function(_vortex_resolve_native_target rustc workspace_root output)
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
    # native static-link manifest rather than an inferred architecture spelling.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(_architecture STREQUAL "x86_64")
            set(_expected_target "x86_64-unknown-linux-gnu")
        elseif(_architecture STREQUAL "aarch64")
            set(_expected_target "aarch64-unknown-linux-gnu")
        else()
            message(FATAL_ERROR
                "Vortex supports native Linux x86_64 and aarch64 only; CMake selected ${_processor}")
        endif()

    elseif(APPLE)
        if(_architecture STREQUAL "x86_64")
            set(_expected_target "x86_64-apple-darwin")
        elseif(_architecture STREQUAL "aarch64")
            set(_expected_target "aarch64-apple-darwin")
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

    set(_rust_target "${_expected_target}")
    if(NOT VORTEX_RESOLVED_RUSTC_HOST STREQUAL _rust_target)
        message(FATAL_ERROR
            "Vortex supports native Rust builds only: rustc host is "
            "${VORTEX_RESOLVED_RUSTC_HOST}, but CMake requires "
            "${_rust_target}")
    endif()

    # Confirm that the selected toolchain contains the target standard library.
    execute_process(
        COMMAND "${rustc}" --print target-libdir --target "${_rust_target}"
        WORKING_DIRECTORY "${workspace_root}"
        OUTPUT_VARIABLE _target_libdir
        ERROR_VARIABLE _target_libdir_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _target_libdir_result)
    if(NOT _target_libdir_result EQUAL 0 OR NOT IS_DIRECTORY "${_target_libdir}")
        message(FATAL_ERROR
            "The Rust standard library for ${_rust_target} is unavailable: "
            "${_target_libdir_error}. Install or provide that target before "
            "configuring Vortex.")
    endif()
    file(GLOB _target_std_rlibs "${_target_libdir}/libstd-*.rlib")
    if(NOT _target_std_rlibs)
        message(FATAL_ERROR
            "The Rust target directory ${_target_libdir} does not contain "
            "libstd for ${_rust_target}. Install or provide that target before "
            "configuring Vortex.")
    endif()

    # Cargo build scripts may invoke either native driver. Validate their
    # effective target tuples against the architecture selected above.
    _vortex_optional_value(CMAKE_C_COMPILER_ARG1 _c_compiler_arg1)
    _vortex_optional_value(CMAKE_C_COMPILER_TARGET _c_compiler_target)
    _vortex_validate_native_compiler(
        "${CMAKE_C_COMPILER}"
        "${_c_compiler_arg1}"
        "${_c_compiler_target}"
        "${CMAKE_C_COMPILER_ID}"
        "${_architecture}"
        "${_rust_target}"
        "C")
    _vortex_optional_value(CMAKE_CXX_COMPILER _cxx_compiler)
    if(_cxx_compiler)
        _vortex_optional_value(CMAKE_CXX_COMPILER_ARG1 _cxx_compiler_arg1)
        _vortex_optional_value(CMAKE_CXX_COMPILER_TARGET _cxx_compiler_target)
        _vortex_validate_native_compiler(
            "${_cxx_compiler}"
            "${_cxx_compiler_arg1}"
            "${_cxx_compiler_target}"
            "${CMAKE_CXX_COMPILER_ID}"
            "${_architecture}"
            "${_rust_target}"
            "C++")
    endif()

    set(${output} "${_rust_target}" PARENT_SCOPE)
endfunction()
