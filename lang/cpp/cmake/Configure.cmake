# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Configures the Rust FFI library used by the Vortex C++ target. This module
# defines the Cargo build and exposes its output to the CMake build.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/Helpers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/RustToolchain.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/SystemDependencies.cmake")

# Validate the single-config CMake build type and return its uppercase CMake
# configuration, Cargo profile, and Cargo artifact directory. Unsupported build
# types and multi-config generators are fatal.
function(_vortex_resolve_cargo_profile configuration_output profile_output artifact_directory_output)
    if(CMAKE_CONFIGURATION_TYPES)
        message(FATAL_ERROR
            "The initial Vortex CMake integration supports single-config "
            "generators only; use Ninja with CMAKE_BUILD_TYPE=Debug, Release, "
            "or RelWithDebInfo")
    endif()

    string(TOUPPER "${CMAKE_BUILD_TYPE}" _configuration)
    if(_configuration STREQUAL "DEBUG")
        set(_cargo_profile "dev")
        set(_artifact_directory "debug")
    elseif(_configuration STREQUAL "RELEASE")
        set(_cargo_profile "release")
        set(_artifact_directory "release")
    elseif(_configuration STREQUAL "RELWITHDEBINFO")
        set(_cargo_profile "release_debug")
        set(_artifact_directory "release_debug")
    else()
        message(FATAL_ERROR
            "Vortex requires CMAKE_BUILD_TYPE=Debug, Release, or "
            "RelWithDebInfo; got '${CMAKE_BUILD_TYPE}'")
    endif()

    set(${configuration_output} "${_configuration}" PARENT_SCOPE)
    set(${profile_output} "${_cargo_profile}" PARENT_SCOPE)
    set(${artifact_directory_output} "${_artifact_directory}" PARENT_SCOPE)
endfunction()

# Select the CPU or CUDA FFI package from a complete workspace checkout. Return
# its package and archive names, public header directories, optional nvcc path,
# and CUDA root;
# missing workspace files and CUDA on non-Linux platforms are fatal.
function(_vortex_resolve_ffi_package
    workspace_root
    package_output
    archive_name_output
    include_dirs_output
    nvcc_output
    cuda_root_output)
    set(_package "vortex-ffi")
    set(_archive_name "libvortex_ffi.a")
    set(_manifest "${workspace_root}/vortex-ffi/Cargo.toml")
    set(_include_dirs "${workspace_root}/vortex-ffi/cinclude")
    set(_nvcc "")
    set(_cuda_root "")

    if(VORTEX_ENABLE_CUDA)
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
            message(FATAL_ERROR "VORTEX_ENABLE_CUDA is supported on Linux only")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(_package "vortex-cuda-ffi")
        set(_archive_name "libvortex_cuda_ffi.a")
        set(_manifest "${workspace_root}/vortex-cuda/ffi/Cargo.toml")
        list(APPEND _include_dirs "${workspace_root}/vortex-cuda/ffi/cinclude")
        set(_nvcc "${CUDAToolkit_NVCC_EXECUTABLE}")
        set(_cuda_root "${CUDAToolkit_TARGET_DIR}")
    endif()

    if(NOT EXISTS "${workspace_root}/Cargo.toml" OR
        NOT EXISTS "${workspace_root}/Cargo.lock" OR
        NOT EXISTS "${_manifest}")
        message(FATAL_ERROR
            "Vortex's CMake source build requires a complete workspace "
            "checkout containing ${_package}")
    endif()

    set(${package_output} "${_package}" PARENT_SCOPE)
    set(${archive_name_output} "${_archive_name}" PARENT_SCOPE)
    set(${include_dirs_output} "${_include_dirs}" PARENT_SCOPE)
    set(${nvcc_output} "${_nvcc}" PARENT_SCOPE)
    set(${cuda_root_output} "${_cuda_root}" PARENT_SCOPE)
endfunction()

# Validate the sanitizer selection against the CMake configuration and resolved
# Rust release. Return the native compiler flag, additional rustc arguments,
# and whether Cargo must rebuild the standard library. Invalid names,
# non-Debug builds, and non-nightly Rust toolchains are fatal.
function(_vortex_resolve_sanitizer
    configuration
    rustc_release
    native_flag_output
    rustflags_output
    build_std_output)
    string(TOLOWER "${VORTEX_SANITIZER}" _sanitizer)
    if(NOT _sanitizer STREQUAL "" AND
        NOT _sanitizer STREQUAL "asan" AND
        NOT _sanitizer STREQUAL "tsan")
        message(FATAL_ERROR
            "VORTEX_SANITIZER must be empty, asan, or tsan; got "
            "${VORTEX_SANITIZER}")
    endif()
    if(NOT _sanitizer STREQUAL "" AND NOT configuration STREQUAL "DEBUG")
        message(FATAL_ERROR "Vortex sanitizer builds require CMAKE_BUILD_TYPE=Debug")
    endif()
    if(NOT _sanitizer STREQUAL "" AND NOT rustc_release MATCHES "nightly")
        message(FATAL_ERROR
            "VORTEX_SANITIZER=${_sanitizer} requires a nightly Rust "
            "toolchain; found ${rustc_release}")
    endif()

    set(_native_flag "")
    set(_rust_sanitizer "")
    if(_sanitizer STREQUAL "asan")
        set(_native_flag "-fsanitize=address,undefined,leak")
        set(_rust_sanitizer "address,leak")
    elseif(_sanitizer STREQUAL "tsan")
        set(_native_flag "-fsanitize=thread")
        set(_rust_sanitizer "thread")
    endif()

    set(_rustflags)
    set(_build_std OFF)
    if(_rust_sanitizer)
        set(_build_std ON)
        list(APPEND _rustflags
            -A warnings
            -Cunsafe-allow-abi-mismatch=sanitizer
            -C debuginfo=2
            -C opt-level=0
            -Zexternal-clangrt
            "-Zsanitizer=${_rust_sanitizer}")
    endif()

    set(${native_flag_output} "${_native_flag}" PARENT_SCOPE)
    set(${rustflags_output} "${_rustflags}" PARENT_SCOPE)
    set(${build_std_output} "${_build_std}" PARENT_SCOPE)
endfunction()

# Resolve the compile and link sysroots that Cargo-built native code and rustc
# must inherit from CMake. Explicit compile/link sysroots take precedence over
# the shared CMake sysroot and the resolved Apple SDK.
function(_vortex_resolve_cargo_sysroots apple_sdkroot compile_output link_output)
    if(CMAKE_SYSROOT_COMPILE)
        set(_compile_sysroot "${CMAKE_SYSROOT_COMPILE}")
    elseif(CMAKE_SYSROOT)
        set(_compile_sysroot "${CMAKE_SYSROOT}")
    elseif(APPLE)
        set(_compile_sysroot "${apple_sdkroot}")
    else()
        set(_compile_sysroot "")
    endif()

    if(CMAKE_SYSROOT_LINK)
        set(_link_sysroot "${CMAKE_SYSROOT_LINK}")
    elseif(CMAKE_SYSROOT)
        set(_link_sysroot "${CMAKE_SYSROOT}")
    elseif(APPLE)
        set(_link_sysroot "${apple_sdkroot}")
    else()
        set(_link_sysroot "")
    endif()

    set(${compile_output} "${_compile_sysroot}" PARENT_SCOPE)
    set(${link_output} "${_link_sysroot}" PARENT_SCOPE)
endfunction()

# Reconstruct CMake's effective C and C++ flags for Cargo build scripts, then
# append target, sysroot, deployment-target, sanitizer, and PIC requirements.
# Return the effective C compiler target and shell-encoded C/C++ flag strings.
function(_vortex_encode_native_flags
    configuration
    compile_sysroot
    apple_deployment_target
    sanitizer_flag
    c_target_output
    cflags_output
    cxxflags_output)
    if(CMAKE_C_COMPILER_TARGET AND CMAKE_C_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        set(_c_target "${CMAKE_C_COMPILER_TARGET}")
    else()
        set(_c_target "")
    endif()
    if(CMAKE_CXX_COMPILER_TARGET AND CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        set(_cxx_target "${CMAKE_CXX_COMPILER_TARGET}")
    else()
        set(_cxx_target "")
    endif()

    set(_cmake_c_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${configuration}}")
    set(_cmake_cxx_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${configuration}}")
    _vortex_reject_semicolon("effective CMAKE_C_FLAGS" "${_cmake_c_flags}")
    _vortex_reject_semicolon("effective CMAKE_CXX_FLAGS" "${_cmake_cxx_flags}")
    separate_arguments(_cflags UNIX_COMMAND "${_cmake_c_flags}")
    separate_arguments(_cxxflags UNIX_COMMAND "${_cmake_cxx_flags}")

    if(compile_sysroot)
        list(APPEND _cflags "--sysroot=${compile_sysroot}")
        list(APPEND _cxxflags "--sysroot=${compile_sysroot}")
    endif()
    if(_c_target)
        list(APPEND _cflags "--target=${_c_target}")
    endif()
    if(_cxx_target)
        list(APPEND _cxxflags "--target=${_cxx_target}")
    endif()
    if(apple_deployment_target)
        list(APPEND _cflags "-mmacosx-version-min=${apple_deployment_target}")
        list(APPEND _cxxflags "-mmacosx-version-min=${apple_deployment_target}")
    endif()
    if(sanitizer_flag)
        list(APPEND _cflags "${sanitizer_flag}")
        list(APPEND _cxxflags "${sanitizer_flag}")
    endif()

    # Native dependencies become part of the archive embedded in shared parents.
    list(APPEND _cflags -fPIC)
    list(APPEND _cxxflags -fPIC)
    _vortex_encode_shell_arguments(_cflags_shell ${_cflags})
    _vortex_encode_shell_arguments(_cxxflags_shell ${_cxxflags})

    set(${c_target_output} "${_c_target}" PARENT_SCOPE)
    set(${cflags_output} "${_cflags_shell}" PARENT_SCOPE)
    set(${cxxflags_output} "${_cxxflags_shell}" PARENT_SCOPE)
endfunction()

# Add PIC, link-sysroot, and Clang-target requirements to optional sanitizer
# rustflags, then encode the exact rustc argv with Cargo's ASCII unit separator.
function(_vortex_encode_rustflags output link_sysroot c_compiler_target)
    set(_rustflags ${ARGN})
    list(APPEND _rustflags
        -C force-frame-pointers=yes
        -C relocation-model=pic)
    if(link_sysroot)
        list(APPEND _rustflags -C "link-arg=--sysroot=${link_sysroot}")
    endif()
    if(c_compiler_target)
        list(APPEND _rustflags -C "link-arg=--target=${c_compiler_target}")
    endif()

    string(ASCII 31 _separator)
    string(JOIN "${_separator}" _encoded ${_rustflags})
    set(${output} "${_encoded}" PARENT_SCOPE)
endfunction()

# Generate compiler wrappers and flag payload files under the CMake-local support
# directory. Return the compiler paths Cargo build scripts must use;
# unchanged support files retain their timestamps.
function(_vortex_prepare_cargo_support
    support_dir
    encoded_rustflags
    cflags
    cxxflags
    c_compiler_output
    cxx_compiler_output)
    file(MAKE_DIRECTORY "${support_dir}")

    _vortex_make_compiler_wrapper(
        "${CMAKE_C_COMPILER}" "${CMAKE_C_COMPILER_ARG1}"
        "${support_dir}" cc _c_compiler)
    _vortex_make_compiler_wrapper(
        "${CMAKE_CXX_COMPILER}" "${CMAKE_CXX_COMPILER_ARG1}"
        "${support_dir}" cxx _cxx_compiler)
    _vortex_write_if_different("${support_dir}/rustflags" "${encoded_rustflags}")
    _vortex_write_if_different("${support_dir}/cflags" "${cflags}")
    _vortex_write_if_different("${support_dir}/cxxflags" "${cxxflags}")

    set(${c_compiler_output} "${_c_compiler}" PARENT_SCOPE)
    set(${cxx_compiler_output} "${_cxx_compiler}" PARENT_SCOPE)
endfunction()

# Orchestrate one source-local Cargo build and expose its staged archive as the
# private dependency consumed by the public C++ target.
block(SCOPE_FOR VARIABLES)
    _vortex_resolve_cargo_profile(_configuration _cargo_profile _cargo_artifact_directory)

    get_filename_component(_workspace_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    _vortex_resolve_ffi_package(
        "${_workspace_root}"
        _ffi_package
        _cargo_archive_name
        _ffi_include_dirs
        _nvcc_executable
        _cuda_root)

    _vortex_find_rust_tools("${_workspace_root}")
    _vortex_resolve_sanitizer(
        "${_configuration}"
        "${VORTEX_RESOLVED_RUSTC_RELEASE}"
        _sanitizer_compile_flag
        _sanitizer_rustflags
        _cargo_build_std)
    _vortex_resolve_native_target(_rust_target)
    _vortex_resolve_apple_settings(_apple_sdkroot _apple_deployment_target)
    _vortex_resolve_cargo_sysroots("${_apple_sdkroot}" _compile_sysroot _link_sysroot)

    if(NOT "$ENV{CARGO_ENCODED_RUSTFLAGS}" STREQUAL "" OR NOT "$ENV{RUSTFLAGS}" STREQUAL "")
        message(STATUS "Vortex ignores ambient Rust flags in its Cargo build")
    endif()

    _vortex_encode_native_flags(
        "${_configuration}"
        "${_compile_sysroot}"
        "${_apple_deployment_target}"
        "${_sanitizer_compile_flag}"
        _native_c_compiler_target
        _native_c_flags
        _native_cxx_flags)
    _vortex_encode_rustflags(
        _encoded_rustflags
        "${_link_sysroot}"
        "${_native_c_compiler_target}"
        ${_sanitizer_rustflags})

    # Cargo owns incremental invalidation inside this CMake-build-local cache.
    # Registering the directory as additional clean state gives the standard
    # CMake clean target the same effect as `cargo clean --target-dir ...`.
    set(_cargo_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${_cargo_target_dir}")
    set(_cargo_support_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo-support")
    _vortex_prepare_cargo_support(
        "${_cargo_support_dir}"
        "${_encoded_rustflags}"
        "${_native_c_flags}"
        "${_native_cxx_flags}"
        _native_c_compiler
        _native_cxx_compiler)
    set(_cargo_ffi_archive
        "${_cargo_target_dir}/${_rust_target}/${_cargo_artifact_directory}/${_cargo_archive_name}")
    set(_ffi_archive "${CMAKE_CURRENT_BINARY_DIR}/vortex-artifacts/libvortex_ffi.a")

    # The phony target lets Cargo own dependency tracking. Copy-if-different in
    # the driver prevents fresh Cargo checks from forcing downstream relinks.
    add_custom_target(vortex_ffi_cargo_build
        COMMAND "${CMAKE_COMMAND}"
            "-DVORTEX_CARGO_EXECUTABLE=${VORTEX_RESOLVED_CARGO_EXECUTABLE}"
            "-DVORTEX_RUSTC_EXECUTABLE=${VORTEX_RESOLVED_RUSTC_EXECUTABLE}"
            "-DVORTEX_RUST_TARGET=${_rust_target}"
            "-DVORTEX_CARGO_TARGET_DIR=${_cargo_target_dir}"
            "-DVORTEX_CARGO_PROFILE=${_cargo_profile}"
            "-DVORTEX_FFI_PACKAGE=${_ffi_package}"
            "-DVORTEX_CARGO_FFI_ARCHIVE=${_cargo_ffi_archive}"
            "-DVORTEX_CARGO_SUPPORT_DIR=${_cargo_support_dir}"
            "-DVORTEX_NVCC_EXECUTABLE=${_nvcc_executable}"
            "-DVORTEX_CUDA_ROOT=${_cuda_root}"
            "-DVORTEX_CARGO_BUILD_STD=${_cargo_build_std}"
            "-DVORTEX_CMAKE_FFI_ARCHIVE=${_ffi_archive}"
            "-DVORTEX_C_COMPILER=${_native_c_compiler}"
            "-DVORTEX_CXX_COMPILER=${_native_cxx_compiler}"
            "-DVORTEX_AR=${CMAKE_AR}"
            "-DVORTEX_RANLIB=${CMAKE_RANLIB}"
            "-DVORTEX_APPLE_DEPLOYMENT_TARGET=${_apple_deployment_target}"
            "-DVORTEX_APPLE_SDKROOT=${_apple_sdkroot}"
            -P "${CMAKE_CURRENT_LIST_DIR}/CargoBuild.cmake"
        BYPRODUCTS "${_ffi_archive}"
        COMMENT "Building the PIC Vortex FFI static archive with Cargo"
        USES_TERMINAL
        VERBATIM)

    # The imported target remains directory-scoped and is only carried to users
    # through Vortex::cpp_static.
    add_library(vortex_ffi_static STATIC IMPORTED)
    set_target_properties(vortex_ffi_static PROPERTIES
        IMPORTED_LOCATION "${_ffi_archive}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ffi_include_dirs}")
    add_dependencies(vortex_ffi_static vortex_ffi_cargo_build)
    _vortex_attach_system_dependencies(vortex_ffi_static "${_rust_target}")
    if(_sanitizer_compile_flag)
        target_compile_options(vortex_ffi_static INTERFACE "${_sanitizer_compile_flag}")
        target_link_options(vortex_ffi_static INTERFACE "${_sanitizer_compile_flag}")
    endif()

    message(STATUS "Vortex Rust target: ${_rust_target}")
    message(STATUS "Vortex Cargo target directory: ${_cargo_target_dir}")
endblock()
