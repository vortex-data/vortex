# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Runs one RustToolchain.cmake scenario in an isolated CMake process.

cmake_minimum_required(VERSION 3.28)

foreach(_required IN ITEMS
    TEST_BINARY_DIR
    TEST_OPERATION
    VORTEX_CPP_SOURCE_DIR)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "RustToolchainCase.cmake requires ${_required}")
    endif()
endforeach()

include("${VORTEX_CPP_SOURCE_DIR}/cmake/RustToolchain.cmake")

function(_make_fake_compiler output)
    set(_compiler "${TEST_BINARY_DIR}/compiler")
    file(WRITE "${_compiler}"
        "#!/bin/sh\nprintf '%s\\n' '${TEST_COMPILER_OUTPUT}'\n")
    file(CHMOD "${_compiler}"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE)
    set(${output} "${_compiler}" PARENT_SCOPE)
endfunction()

if(TEST_OPERATION STREQUAL "VALIDATE_COMPILER")
    _make_fake_compiler(_compiler)
    set(CMAKE_SYSTEM_NAME "${TEST_SYSTEM_NAME}")
    set(CMAKE_OSX_ARCHITECTURES "${TEST_OSX_ARCHITECTURES}")
    set(CMAKE_C_COMPILER "${_compiler}")
    _vortex_validate_native_compiler(C "${TEST_RUST_TARGET}")

elseif(TEST_OPERATION STREQUAL "RESOLVE_NATIVE_TARGET")
    _make_fake_compiler(_compiler)
    set(CMAKE_CROSSCOMPILING OFF)
    set(CMAKE_SYSTEM_NAME "${TEST_SYSTEM_NAME}")
    set(CMAKE_SYSTEM_PROCESSOR "${TEST_PROCESSOR}")
    set(CMAKE_OSX_ARCHITECTURES "${TEST_OSX_ARCHITECTURES}")
    set(CMAKE_SYSROOT "${TEST_SYSROOT}")
    set(CMAKE_SYSROOT_COMPILE "${TEST_SYSROOT_COMPILE}")
    set(CMAKE_C_COMPILER "${_compiler}")
    set(CMAKE_CXX_COMPILER "${_compiler}")
    set(VORTEX_RESOLVED_RUSTC_HOST "${TEST_RUST_HOST}")

    _vortex_resolve_native_target(_rust_target)
    if(NOT "${TEST_EXPECTED_TARGET}" STREQUAL "" AND
        NOT "${_rust_target}" STREQUAL "${TEST_EXPECTED_TARGET}")
        message(FATAL_ERROR
            "Expected Rust target ${TEST_EXPECTED_TARGET}; got ${_rust_target}")
    endif()

elseif(TEST_OPERATION STREQUAL "RESOLVE_PROGRAM")
    set(_tool_dir "${TEST_BINARY_DIR}/bin")
    set(_tool "${_tool_dir}/cargo")
    file(MAKE_DIRECTORY "${_tool_dir}")
    file(WRITE "${_tool}" "#!/bin/sh\nprintf '%s\\n' 'cargo 99.0.0'\n")
    file(CHMOD "${_tool}"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE)

    # An inherited find_program result must not bypass the local lookup.
    set(CMAKE_PROGRAM_PATH "${_tool_dir}")
    set(_program "/bin/false")
    _vortex_resolve_rust_program("${VORTEX_CPP_SOURCE_DIR}" cargo _resolved_tool)
    file(REAL_PATH "${_tool}" _expected_tool)
    if(NOT "${_resolved_tool}" STREQUAL "${_expected_tool}")
        message(FATAL_ERROR
            "Expected Cargo at ${_expected_tool}; got ${_resolved_tool}")
    endif()

elseif(TEST_OPERATION STREQUAL "RESOLVE_APPLE_SETTINGS")
    set(CMAKE_SYSTEM_NAME "Darwin")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${TEST_OSX_DEPLOYMENT_TARGET}")
    if(TEST_SDK_KIND STREQUAL "macos")
        set(CMAKE_OSX_SYSROOT "${TEST_BINARY_DIR}/MacOSX14.4.sdk")
        file(MAKE_DIRECTORY "${CMAKE_OSX_SYSROOT}")
    elseif(TEST_SDK_KIND STREQUAL "iphone")
        set(CMAKE_OSX_SYSROOT "${TEST_BINARY_DIR}/iPhoneOS17.4.sdk")
        file(MAKE_DIRECTORY "${CMAKE_OSX_SYSROOT}")
    else()
        set(CMAKE_OSX_SYSROOT "")
    endif()

    _make_fake_compiler(_compiler)
    set(CMAKE_CXX_COMPILER "${_compiler}")
    _vortex_resolve_apple_settings(_sdkroot _deployment_target)
    if(NOT "${TEST_EXPECTED_DEPLOYMENT}" STREQUAL "" AND
        NOT "${_deployment_target}" STREQUAL "${TEST_EXPECTED_DEPLOYMENT}")
        message(FATAL_ERROR
            "Expected macOS deployment target ${TEST_EXPECTED_DEPLOYMENT}; "
            "got ${_deployment_target}")
    endif()

else()
    message(FATAL_ERROR "Unknown Rust toolchain test operation: ${TEST_OPERATION}")
endif()
