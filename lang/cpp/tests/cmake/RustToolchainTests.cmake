# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Exercises Rust tool lookup and native ABI validation without building Cargo.

cmake_minimum_required(VERSION 3.28)

foreach(_required IN ITEMS
    VORTEX_CPP_SOURCE_DIR
    VORTEX_TEST_BINARY_DIR)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "RustToolchainTests.cmake requires ${_required}")
    endif()
endforeach()

set(_case_script "${CMAKE_CURRENT_LIST_DIR}/RustToolchainCase.cmake")

function(_run_case name)
    set(_one_value_arguments
        COMPILER_OUTPUT
        EXPECTED_DEPLOYMENT
        EXPECTED_ERROR
        EXPECTED_TARGET
        OPERATION
        OSX_ARCHITECTURES
        OSX_DEPLOYMENT_TARGET
        PROCESSOR
        RUST_HOST
        RUST_TARGET
        SDK_KIND
        SYSTEM_NAME
        SYSROOT
        SYSROOT_COMPILE)
    cmake_parse_arguments(PARSE_ARGV 1 CASE "" "${_one_value_arguments}" "")
    if(CASE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Unexpected arguments for ${name}: ${CASE_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT CASE_OPERATION)
        message(FATAL_ERROR "Toolchain test ${name} requires OPERATION")
    endif()

    set(_binary_dir "${VORTEX_TEST_BINARY_DIR}/${name}")
    file(REMOVE_RECURSE "${_binary_dir}")
    file(MAKE_DIRECTORY "${_binary_dir}")
    set(_command
        "${CMAKE_COMMAND}"
        "-DTEST_BINARY_DIR=${_binary_dir}"
        "-DTEST_COMPILER_OUTPUT=${CASE_COMPILER_OUTPUT}"
        "-DTEST_EXPECTED_DEPLOYMENT=${CASE_EXPECTED_DEPLOYMENT}"
        "-DTEST_EXPECTED_TARGET=${CASE_EXPECTED_TARGET}"
        "-DTEST_OPERATION=${CASE_OPERATION}"
        "-DTEST_OSX_ARCHITECTURES=${CASE_OSX_ARCHITECTURES}"
        "-DTEST_OSX_DEPLOYMENT_TARGET=${CASE_OSX_DEPLOYMENT_TARGET}"
        "-DTEST_PROCESSOR=${CASE_PROCESSOR}"
        "-DTEST_RUST_HOST=${CASE_RUST_HOST}"
        "-DTEST_RUST_TARGET=${CASE_RUST_TARGET}"
        "-DTEST_SDK_KIND=${CASE_SDK_KIND}"
        "-DTEST_SYSTEM_NAME=${CASE_SYSTEM_NAME}"
        "-DTEST_SYSROOT=${CASE_SYSROOT}"
        "-DTEST_SYSROOT_COMPILE=${CASE_SYSROOT_COMPILE}"
        "-DVORTEX_CPP_SOURCE_DIR=${VORTEX_CPP_SOURCE_DIR}"
        -P "${_case_script}")
    execute_process(
        COMMAND ${_command}
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
        RESULT_VARIABLE _result)
    string(CONCAT _log "${_output}" "${_error}")

    if(CASE_EXPECTED_ERROR)
        if(_result EQUAL 0)
            message(FATAL_ERROR
                "Toolchain test ${name} unexpectedly succeeded:\n${_log}")
        elseif(NOT _log MATCHES "${CASE_EXPECTED_ERROR}")
            message(FATAL_ERROR
                "Toolchain test ${name} failed for the wrong reason. Expected "
                "'${CASE_EXPECTED_ERROR}' in:\n${_log}")
        endif()
    elseif(NOT _result EQUAL 0)
        message(FATAL_ERROR "Toolchain test ${name} failed:\n${_log}")
    endif()

    message(STATUS "Passed Rust toolchain case: ${name}")
endfunction()

_run_case(linux_x86_64
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT x86_64-linux-gnu
    RUST_TARGET x86_64-unknown-linux-gnu)
_run_case(linux_aarch64
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT aarch64-unknown-linux-gnu
    RUST_TARGET aarch64-unknown-linux-gnu)
_run_case(linux_vendor
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT x86_64-redhat-linux
    RUST_TARGET x86_64-unknown-linux-gnu)
_run_case(macos_darwin
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT arm64-apple-darwin25.6.0
    RUST_TARGET aarch64-apple-darwin)
_run_case(macos_macosx
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT arm64-apple-macosx14.0
    RUST_TARGET aarch64-apple-darwin)

foreach(_target IN ITEMS
    x86_64-linux-gnux32
    x86_64-unknown-linux-musl
    x86_64-linux-android
    x86_64-linux-uclibc)
    _run_case(reject_${_target}
        OPERATION VALIDATE_COMPILER
        COMPILER_OUTPUT "${_target}"
        RUST_TARGET x86_64-unknown-linux-gnu
        EXPECTED_ERROR "Linux ABI")
endforeach()
_run_case(reject_ios
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT arm64-apple-ios17.0
    RUST_TARGET aarch64-apple-darwin
    EXPECTED_ERROR "does not use the macOS")
_run_case(reject_catalyst
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT arm64-apple-ios13.1-macabi
    RUST_TARGET aarch64-apple-darwin
    EXPECTED_ERROR "does not use the macOS")
_run_case(reject_architecture_mismatch
    OPERATION VALIDATE_COMPILER
    COMPILER_OUTPUT aarch64-linux-gnu
    RUST_TARGET x86_64-unknown-linux-gnu
    EXPECTED_ERROR "does not match Vortex Rust target")

_run_case(resolve_linux_target
    OPERATION RESOLVE_NATIVE_TARGET
    COMPILER_OUTPUT x86_64-linux-gnu
    SYSTEM_NAME Linux
    PROCESSOR amd64
    RUST_HOST x86_64-unknown-linux-gnu
    EXPECTED_TARGET x86_64-unknown-linux-gnu)
_run_case(resolve_macos_target
    OPERATION RESOLVE_NATIVE_TARGET
    COMPILER_OUTPUT arm64-apple-darwin25.6.0
    SYSTEM_NAME Darwin
    PROCESSOR arm64
    OSX_ARCHITECTURES arm64
    RUST_HOST aarch64-apple-darwin
    EXPECTED_TARGET aarch64-apple-darwin)
_run_case(reject_explicit_macos_x86_64
    OPERATION RESOLVE_NATIVE_TARGET
    SYSTEM_NAME Darwin
    PROCESSOR x86_64
    OSX_ARCHITECTURES x86_64
    RUST_HOST x86_64-apple-darwin
    EXPECTED_ERROR "must be exactly arm64")
_run_case(reject_implicit_macos_x86_64
    OPERATION RESOLVE_NATIVE_TARGET
    SYSTEM_NAME Darwin
    PROCESSOR x86_64
    RUST_HOST x86_64-apple-darwin
    EXPECTED_ERROR "supports arm64 only")
_run_case(reject_compile_sysroot
    OPERATION RESOLVE_NATIVE_TARGET
    SYSTEM_NAME Linux
    PROCESSOR x86_64
    RUST_HOST x86_64-unknown-linux-gnu
    SYSROOT_COMPILE /tmp/vortex-sysroot
    EXPECTED_ERROR "CMAKE_SYSROOT or CMAKE_SYSROOT_COMPILE")
_run_case(reject_non_macos_apple_platform
    OPERATION RESOLVE_NATIVE_TARGET
    SYSTEM_NAME iOS
    PROCESSOR arm64
    RUST_HOST aarch64-apple-darwin
    EXPECTED_ERROR "CMake selected iOS")

_run_case(ignore_inherited_find_program_result
    OPERATION RESOLVE_PROGRAM)
_run_case(resolve_macos_sdk
    OPERATION RESOLVE_APPLE_SETTINGS
    SDK_KIND macos
    OSX_DEPLOYMENT_TARGET 14.0
    EXPECTED_DEPLOYMENT 14.0)
_run_case(reject_iphone_sdk
    OPERATION RESOLVE_APPLE_SETTINGS
    SDK_KIND iphone
    OSX_DEPLOYMENT_TARGET 17.0
    EXPECTED_ERROR "must resolve to a macOS SDK")
_run_case(resolve_legacy_deployment_target
    OPERATION RESOLVE_APPLE_SETTINGS
    COMPILER_OUTPUT "#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 1090"
    EXPECTED_DEPLOYMENT 10.9)
_run_case(resolve_current_deployment_target
    OPERATION RESOLVE_APPLE_SETTINGS
    COMPILER_OUTPUT "#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 101203"
    EXPECTED_DEPLOYMENT 10.12.3)
