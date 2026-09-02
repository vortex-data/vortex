# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Build-time `cmake -P` driver for the custom target defined by
# VortexCargo.cmake. Configuration arrives through -D variables; this script
# defines no CMake targets. It recreates the validated Cargo environment and
# stages the resulting archive at the stable path consumed by CMake. Missing
# inputs, Cargo failure, or a missing/staging-failed archive are fatal.

cmake_minimum_required(VERSION 3.28)

# These non-empty inputs must cross from configure time as -D definitions. Fail
# before invoking Cargo rather than falling back to ambient build settings.
foreach(_required IN ITEMS
    VORTEX_CARGO_EXECUTABLE
    VORTEX_RUSTC_EXECUTABLE
    VORTEX_RUST_TARGET
    VORTEX_WORKSPACE_ROOT
    VORTEX_FFI_MANIFEST
    VORTEX_CARGO_CONFIG_FILE
    VORTEX_CARGO_TARGET_DIR
    VORTEX_CARGO_PROFILE
    VORTEX_CARGO_FFI_ARCHIVE
    VORTEX_CMAKE_FFI_ARCHIVE
    VORTEX_TARGET_ENV_KEY_UPPER
    VORTEX_TARGET_ENV_KEY_LOWER
    VORTEX_RUSTFLAGS_FILE
    VORTEX_CFLAGS_FILE
    VORTEX_CXXFLAGS_FILE)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "BuildVortexCargo.cmake requires ${_required}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${VORTEX_CARGO_TARGET_DIR}")

# Build commands are CMake lists so every option and value remains a distinct
# argv element. The calling custom target uses VERBATIM for its outer cmake -P
# invocation; execute_process consumes this inner list without a shell string.
set(_cargo_command "${VORTEX_CARGO_EXECUTABLE}" --config "${VORTEX_CARGO_CONFIG_FILE}")
if(VORTEX_CARGO_OFFLINE)
    list(APPEND _cargo_command --offline)
endif()
list(APPEND _cargo_command
    rustc
    --locked
    --package vortex-ffi
    --lib
    --crate-type=staticlib
    --manifest-path "${VORTEX_FFI_MANIFEST}"
    --target "${VORTEX_RUST_TARGET}"
    --target-dir "${VORTEX_CARGO_TARGET_DIR}"
    --profile "${VORTEX_CARGO_PROFILE}")
if(DEFINED VORTEX_CARGO_JOBS AND NOT VORTEX_CARGO_JOBS STREQUAL "")
    list(APPEND _cargo_command --jobs "${VORTEX_CARGO_JOBS}")
endif()
if(VORTEX_CARGO_NO_DEFAULT_FEATURES)
    list(APPEND _cargo_command --no-default-features)
endif()
if(DEFINED VORTEX_CARGO_FEATURES AND NOT VORTEX_CARGO_FEATURES STREQUAL "")
    list(APPEND _cargo_command --features "${VORTEX_CARGO_FEATURES}")
endif()
if(VORTEX_CARGO_BUILD_STD)
    list(APPEND _cargo_command -Zbuild-std)
endif()

# Flag payloads use files because shell-quoted native flags and ASCII-31 encoded
# Rust flags cannot safely traverse another -D and CMake-list expansion.
foreach(_flags_file IN ITEMS VORTEX_RUSTFLAGS_FILE VORTEX_CFLAGS_FILE VORTEX_CXXFLAGS_FILE)
    if(NOT EXISTS "${${_flags_file}}")
        message(FATAL_ERROR "BuildVortexCargo.cmake cannot read ${_flags_file}: ${${_flags_file}}")
    endif()
endforeach()
file(READ "${VORTEX_RUSTFLAGS_FILE}" _encoded_rustflags)
file(READ "${VORTEX_CFLAGS_FILE}" _native_c_flags)
file(READ "${VORTEX_CXXFLAGS_FILE}" _native_cxx_flags)

# Cargo and its subprocesses must use the concrete rustc selected at configure
# time, not a rustup proxy or another PATH entry. Put its bin directory first
# and set both RUSTC and CARGO_BUILD_RUSTC below.
get_filename_component(_rust_toolchain_bin_dir "${VORTEX_RUSTC_EXECUTABLE}" DIRECTORY)
if(DEFINED ENV{PATH} AND NOT "$ENV{PATH}" STREQUAL "")
    if("$ENV{PATH}" MATCHES ";")
        message(FATAL_ERROR "The CMake-owned Cargo build does not support semicolons in PATH")
    endif()
    set(_cargo_path "${_rust_toolchain_bin_dir}:$ENV{PATH}")
else()
    set(_cargo_path "${_rust_toolchain_bin_dir}")
endif()

# Target-qualified linker, CC/CXX, AR, RANLIB, and flag variables force Cargo
# build scripts and native dependencies through CMake's selected toolchain while
# leaving unrelated host-target settings untouched.
set(_environment
    "PATH=${_cargo_path}"
    "RUSTC=${VORTEX_RUSTC_EXECUTABLE}"
    "CARGO_BUILD_RUSTC=${VORTEX_RUSTC_EXECUTABLE}"
    "CC_SHELL_ESCAPED_FLAGS=1"
    "CARGO_TARGET_${VORTEX_TARGET_ENV_KEY_UPPER}_LINKER=${VORTEX_C_LINKER}"
    "CC_${VORTEX_TARGET_ENV_KEY_LOWER}=${VORTEX_C_COMPILER}"
    "CXX_${VORTEX_TARGET_ENV_KEY_LOWER}=${VORTEX_CXX_COMPILER}")
if(DEFINED VORTEX_AR AND NOT VORTEX_AR STREQUAL "")
    list(APPEND _environment "AR_${VORTEX_TARGET_ENV_KEY_LOWER}=${VORTEX_AR}")
endif()
if(DEFINED VORTEX_RANLIB AND NOT VORTEX_RANLIB STREQUAL "")
    list(APPEND _environment "RANLIB_${VORTEX_TARGET_ENV_KEY_LOWER}=${VORTEX_RANLIB}")
endif()
if(_native_c_flags)
    list(APPEND _environment "CFLAGS_${VORTEX_TARGET_ENV_KEY_LOWER}=${_native_c_flags}")
endif()
if(_native_cxx_flags)
    list(APPEND _environment "CXXFLAGS_${VORTEX_TARGET_ENV_KEY_LOWER}=${_native_cxx_flags}")
endif()
if(DEFINED VORTEX_APPLE_DEPLOYMENT_TARGET AND NOT VORTEX_APPLE_DEPLOYMENT_TARGET STREQUAL "")
    list(APPEND _environment "MACOSX_DEPLOYMENT_TARGET=${VORTEX_APPLE_DEPLOYMENT_TARGET}")
endif()
if(DEFINED VORTEX_APPLE_SDKROOT AND NOT VORTEX_APPLE_SDKROOT STREQUAL "")
    list(APPEND _environment "SDKROOT=${VORTEX_APPLE_SDKROOT}")
endif()

# Use one validated, fingerprinted flag sequence at Cargo's highest-precedence
# environment layer. The explicit unsets remove ambient global and target Rust
# flags before installing that sequence, preventing hidden target-CPU or PIC
# changes from bypassing configure-time validation.
list(APPEND _environment "CARGO_ENCODED_RUSTFLAGS=${_encoded_rustflags}")
set(_command
    "${CMAKE_COMMAND}" -E env
    --unset=RUSTFLAGS
    --unset=CARGO_ENCODED_RUSTFLAGS
    "--unset=CARGO_TARGET_${VORTEX_TARGET_ENV_KEY_UPPER}_RUSTFLAGS"
    ${_environment}
    ${_cargo_command})
# The caller's always-out-of-date custom target runs Cargo on every build.
# Cargo's own dependency cache decides whether compilation work is necessary.
execute_process(
    COMMAND ${_command}
    WORKING_DIRECTORY "${VORTEX_WORKSPACE_ROOT}"
    COMMAND_ECHO STDOUT
    RESULT_VARIABLE _cargo_result)
if(NOT _cargo_result EQUAL 0)
    message(FATAL_ERROR "Cargo failed while building vortex-ffi (${_cargo_result})")
endif()

if(NOT EXISTS "${VORTEX_CARGO_FFI_ARCHIVE}")
    message(FATAL_ERROR
        "Cargo completed successfully but did not produce the expected "
        "static archive: ${VORTEX_CARGO_FFI_ARCHIVE}")
endif()

get_filename_component(_cmake_archive_directory "${VORTEX_CMAKE_FFI_ARCHIVE}" DIRECTORY)
file(MAKE_DIRECTORY "${_cmake_archive_directory}")
# Stage only changed bytes so repeated no-op Cargo runs do not advance the
# imported archive's timestamp and trigger avoidable downstream relinks.
file(COPY_FILE "${VORTEX_CARGO_FFI_ARCHIVE}" "${VORTEX_CMAKE_FFI_ARCHIVE}" ONLY_IF_DIFFERENT)
if(NOT EXISTS "${VORTEX_CMAKE_FFI_ARCHIVE}")
    message(FATAL_ERROR "Failed to stage the vortex-ffi archive for CMake: ${VORTEX_CMAKE_FFI_ARCHIVE}")
endif()
