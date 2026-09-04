# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Builds and stages the Rust FFI static library used by the Vortex C++ target.
# Configure.cmake invokes this internal script automatically during the build.

cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/Helpers.cmake")

# Configure.cmake passes these required values as `-D` arguments when the
# `vortex_ffi_cargo_build` target launches this script with `cmake -P`.
function(_vortex_require_build_inputs)
    foreach(_required IN ITEMS
        VORTEX_CARGO_EXECUTABLE
        VORTEX_RUSTC_EXECUTABLE
        VORTEX_RUST_TARGET
        VORTEX_CARGO_TARGET_DIR
        VORTEX_CARGO_PROFILE
        VORTEX_FFI_PACKAGE
        VORTEX_CARGO_FFI_ARCHIVE
        VORTEX_CMAKE_FFI_ARCHIVE
        VORTEX_RUSTFLAGS
        VORTEX_CFLAGS
        VORTEX_CXXFLAGS
        VORTEX_C_COMPILER
        VORTEX_CXX_COMPILER
        VORTEX_AR
        VORTEX_RANLIB)
        if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
            message(FATAL_ERROR "CargoBuild.cmake requires ${_required}")
        endif()
    endforeach()
endfunction()

# Format the Rust target for `cc` environment variable names.
function(_vortex_cc_target_env_key output)
    string(REPLACE "-" "_" _target_key "${VORTEX_RUST_TARGET}")
    string(TOLOWER "${_target_key}" _target_key)
    set(${output} "${_target_key}" PARENT_SCOPE)
endfunction()

# Assemble the Cargo command that builds the selected FFI package as
# a static library with the selected Cargo profile.
function(_vortex_make_cargo_command output)
    set(_command
        "${VORTEX_CARGO_EXECUTABLE}"
        rustc
        --locked
        --package "${VORTEX_FFI_PACKAGE}"
        --lib
        --crate-type=staticlib
        --target "${VORTEX_RUST_TARGET}"
        --target-dir "${VORTEX_CARGO_TARGET_DIR}"
        --profile "${VORTEX_CARGO_PROFILE}")
    if(VORTEX_CARGO_BUILD_STD)
        list(APPEND _command -Zbuild-std)
    endif()
    set(${output} "${_command}" PARENT_SCOPE)
endfunction()

# Prepend the selected Rust and CUDA toolchain directories to the ambient PATH.
function(_vortex_build_tool_path output)
    get_filename_component(_path "${VORTEX_RUSTC_EXECUTABLE}" DIRECTORY)

    if(VORTEX_NVCC_EXECUTABLE)
        get_filename_component(_nvcc_bin_dir "${VORTEX_NVCC_EXECUTABLE}" DIRECTORY)
        string(APPEND _path ":${_nvcc_bin_dir}")
    endif()

    if("$ENV{PATH}" MATCHES ";")
        message(FATAL_ERROR "The CMake-owned Cargo build does not support semicolons in PATH")
    elseif(NOT "$ENV{PATH}" STREQUAL "")
        string(APPEND _path ":$ENV{PATH}")
    endif()

    set(${output} "${_path}" PARENT_SCOPE)
endfunction()

# Assemble the target-specific Cargo environment from the selected tools and
# flags.
function(_vortex_make_cargo_environment target_key_lower output)
    # Cargo separates CARGO_ENCODED_RUSTFLAGS arguments with ASCII unit separator,
    # and the cc crate reads shell-quoted words with CC_SHELL_ESCAPED_FLAGS.
    string(ASCII 31 _separator)
    string(JOIN "${_separator}" _rustflags ${VORTEX_RUSTFLAGS})
    _vortex_encode_shell_arguments(_cflags ${VORTEX_CFLAGS})
    _vortex_encode_shell_arguments(_cxxflags ${VORTEX_CXXFLAGS})
    _vortex_build_tool_path(_cargo_path)

    set(_environment
        "PATH=${_cargo_path}"
        "RUSTC=${VORTEX_RUSTC_EXECUTABLE}"
        "CC_SHELL_ESCAPED_FLAGS=1"
        "CC_${target_key_lower}=${VORTEX_C_COMPILER}"
        "CXX_${target_key_lower}=${VORTEX_CXX_COMPILER}"
        "AR_${target_key_lower}=${VORTEX_AR}"
        "RANLIB_${target_key_lower}=${VORTEX_RANLIB}"
        "CFLAGS_${target_key_lower}=${_cflags}"
        "CXXFLAGS_${target_key_lower}=${_cxxflags}"
        "CARGO_ENCODED_RUSTFLAGS=${_rustflags}")

    # Keep the toolchain selection seen at configure time, even when the
    # ambient environment differs at build time.
    if(VORTEX_RUSTUP_TOOLCHAIN)
        list(APPEND _environment "RUSTUP_TOOLCHAIN=${VORTEX_RUSTUP_TOOLCHAIN}")
    else()
        list(APPEND _environment --unset=RUSTUP_TOOLCHAIN)
    endif()

    if(VORTEX_CUDA_ROOT)
        list(APPEND _environment "CUDA_PATH=${VORTEX_CUDA_ROOT}")
    endif()

    if(VORTEX_APPLE_DEPLOYMENT_TARGET)
        list(APPEND _environment
            "MACOSX_DEPLOYMENT_TARGET=${VORTEX_APPLE_DEPLOYMENT_TARGET}")
    endif()

    set(${output} "${_environment}" PARENT_SCOPE)
endfunction()

_vortex_require_build_inputs()
get_filename_component(_workspace_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
_vortex_cc_target_env_key(_target_key)
_vortex_make_cargo_command(_cargo_command)
_vortex_make_cargo_environment("${_target_key}" _cargo_environment)

# Run Cargo with the CMake-selected tools and flags. Cargo remains responsible
# for dependency tracking and incremental freshness.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        ${_cargo_environment}
        ${_cargo_command}
    WORKING_DIRECTORY "${_workspace_root}"
    RESULT_VARIABLE _cargo_result)
if(NOT _cargo_result EQUAL 0)
    message(FATAL_ERROR
        "Cargo failed while building ${VORTEX_FFI_PACKAGE} (${_cargo_result})")
endif()

# Catch profile, target, or crate-output changes that invalidate the archive
# path predicted by Configure.cmake.
if(NOT EXISTS "${VORTEX_CARGO_FFI_ARCHIVE}")
    message(FATAL_ERROR
        "Cargo completed successfully but did not produce the expected "
        "static archive: ${VORTEX_CARGO_FFI_ARCHIVE}")
endif()

# Stage the Cargo archive at CMake's stable artifact path. Preserve its
# timestamp when the contents are unchanged to avoid unnecessary relinks.
get_filename_component(_destination_dir "${VORTEX_CMAKE_FFI_ARCHIVE}" DIRECTORY)
file(MAKE_DIRECTORY "${_destination_dir}")
file(COPY_FILE
    "${VORTEX_CARGO_FFI_ARCHIVE}" "${VORTEX_CMAKE_FFI_ARCHIVE}"
    ONLY_IF_DIFFERENT)
