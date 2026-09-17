# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Builds and stages the Rust FFI static library used by the Vortex C++ target.
# Configure.cmake invokes this internal script automatically during the build.

cmake_minimum_required(VERSION 3.25)

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
        VORTEX_FFI_HEADERS
        VORTEX_CMAKE_FFI_INCLUDE_DIR
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

# An explicit native --target isolates library Rust flags from Cargo's build tools.
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

# Put the native launchers and selected toolchains before the ambient PATH.
function(_vortex_build_tool_path output)
    if(VORTEX_CARGO_TARGET_DIR MATCHES ":")
        message(FATAL_ERROR "The Cargo target directory cannot contain ':' because native launchers use PATH")
    endif()
    get_filename_component(_rust_bin_dir "${VORTEX_RUSTC_EXECUTABLE}" DIRECTORY)
    set(_path "${VORTEX_CARGO_TARGET_DIR}/cmake-native-tools:${_rust_bin_dir}")

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

# Wrap the compiler for host-only flag filtering without hiding arguments from cc-rs probes.
function(_vortex_native_compiler_launcher compiler arg1 output)
    _vortex_reject_semicolon("compiler ARG1" "${arg1}")
    _vortex_encode_shell_arguments(_compiler "${compiler}")
    separate_arguments(_command UNIX_COMMAND "${_compiler} ${arg1}")
    # Quote the list expansion to preserve empty mandatory arguments.
    _vortex_encode_shell_arguments(_compiler_command "${_command}")

    # Match cc-rs: an explicit known wrapper suppresses RUSTC_WRAPPER fallback.
    # Keep its command intact rather than caching env or double-wrapping sccache.
    get_filename_component(_compiler_name "${compiler}" NAME_WLE)
    set(_known_wrappers ccache distcc sccache icecc cachepot buildcache kache env
        "$ENV{CC_KNOWN_WRAPPER_CUSTOM}")
    set(_use_rustc_wrapper true)
    list(LENGTH _command _command_length)
    if(_command_length GREATER 1 AND _compiler_name IN_LIST _known_wrappers)
        set(_use_rustc_wrapper false)
    endif()

    file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/NativeCompiler.sh.in" _template)
    string(CONFIGURE "${_template}" _script @ONLY)

    # cc-rs tracks CC/CXX, not launcher contents. Change the path when the
    # compiler, mandatory arguments, or script changes; CFLAGS/CXXFLAGS track flags.
    string(SHA256 _key "${_script}")
    set(_directory "${VORTEX_CARGO_TARGET_DIR}/cmake-native-tools")
    set(_launcher "${_directory}/cc-${_key}")
    file(MAKE_DIRECTORY "${_directory}")
    file(WRITE "${_launcher}" "${_script}")
    file(CHMOD "${_launcher}" PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    # An explicit cc-rs wrapper suppresses its outer RUSTC_WRAPPER fallback.
    # It splits CC/CXX on whitespace, so resolve the launcher through PATH.
    set(${output} "env cc-${_key}" PARENT_SCOPE)
endfunction()

# Assemble the Cargo environment from the selected tools and flags.
function(_vortex_make_cargo_environment output)
    string(REPLACE "-" "_" _target_key "${VORTEX_RUST_TARGET}")
    string(TOLOWER "${_target_key}" _target_key)

    # Cargo separates CARGO_ENCODED_RUSTFLAGS arguments with ASCII unit separator,
    # and the cc crate reads shell-quoted words with CC_SHELL_ESCAPED_FLAGS.
    string(ASCII 31 _separator)
    string(JOIN "${_separator}" _rustflags ${VORTEX_RUSTFLAGS})
    _vortex_encode_shell_arguments(_cflags ${VORTEX_CFLAGS})
    _vortex_encode_shell_arguments(_cxxflags ${VORTEX_CXXFLAGS})
    _vortex_native_compiler_launcher("${VORTEX_C_COMPILER}" "${VORTEX_C_COMPILER_ARG1}" _cc)
    _vortex_native_compiler_launcher("${VORTEX_CXX_COMPILER}" "${VORTEX_CXX_COMPILER_ARG1}" _cxx)
    _vortex_build_tool_path(_cargo_path)

    set(_environment
        "PATH=${_cargo_path}"
        "RUSTC=${VORTEX_RUSTC_EXECUTABLE}"
        "CC_SHELL_ESCAPED_FLAGS=1"
        "CC_KNOWN_WRAPPER_CUSTOM=env"
        "CFLAGS_${_target_key}=${_cflags}"
        "CXXFLAGS_${_target_key}=${_cxxflags}"
        "CARGO_ENCODED_RUSTFLAGS=${_rustflags}")

    # Tool lookup prefers the literal triple over its underscore spelling.
    foreach(_key IN ITEMS "${VORTEX_RUST_TARGET}" "${_target_key}")
        list(APPEND _environment
            "CC_${_key}=${_cc}"
            "CXX_${_key}=${_cxx}"
            "AR_${_key}=${VORTEX_AR}"
            "RANLIB_${_key}=${VORTEX_RANLIB}")
    endforeach()

    # Keep the toolchain selection seen at configure time, even when the
    # ambient environment differs at build time.
    if(VORTEX_RUSTUP_TOOLCHAIN)
        list(APPEND _environment "RUSTUP_TOOLCHAIN=${VORTEX_RUSTUP_TOOLCHAIN}")
    else()
        list(APPEND _environment --unset=RUSTUP_TOOLCHAIN)
    endif()

    if(VORTEX_CUDA_ROOT)
        # Architecture tokens contain no whitespace. Preserve an empty value for OFF;
        # omitting it would inherit ambient flags or fall back to native.
        string(JOIN " " _cuda_arch_flags ${VORTEX_CUDA_ARCH_FLAGS})
        # An empty host compiler clears inherited Vortex overrides without overriding NVCC defaults.
        list(APPEND _environment
            "CUDA_PATH=${VORTEX_CUDA_ROOT}"
            "VORTEX_CUDA_ARCH_FLAGS=${_cuda_arch_flags}"
            "VORTEX_CUDA_HOST_COMPILER=${VORTEX_CUDA_HOST_COMPILER}")
    endif()

    if(VORTEX_APPLE_DEPLOYMENT_TARGET)
        list(APPEND _environment
            "MACOSX_DEPLOYMENT_TARGET=${VORTEX_APPLE_DEPLOYMENT_TARGET}")
    endif()

    set(${output} "${_environment}" PARENT_SCOPE)
endfunction()

_vortex_require_build_inputs()
get_filename_component(_workspace_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
_vortex_make_cargo_command(_cargo_command)
_vortex_make_cargo_environment(_cargo_environment)

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

# Cargo may regenerate checkout headers; publish them before consumers compile.
file(MAKE_DIRECTORY "${VORTEX_CMAKE_FFI_INCLUDE_DIR}")
foreach(_header IN LISTS VORTEX_FFI_HEADERS)
    get_filename_component(_name "${_header}" NAME)
    file(COPY_FILE "${_header}" "${VORTEX_CMAKE_FFI_INCLUDE_DIR}/${_name}"
        ONLY_IF_DIFFERENT)
endforeach()
