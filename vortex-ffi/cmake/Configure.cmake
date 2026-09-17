# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Configures the Rust FFI library used by the Vortex C++ target. This module
# defines the Cargo build and exposes its output to the CMake build.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/Cuda.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/DebugInfo.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/Helpers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/RustToolchain.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/SystemDependencies.cmake")

# Map the single-config build type to a Cargo profile unless explicitly overridden.
function(_vortex_resolve_cargo_profile configuration_output profile_output artifact_directory_output)
    if(CMAKE_CONFIGURATION_TYPES)
        message(FATAL_ERROR
            "The initial Vortex CMake integration supports single-config "
            "generators only; use Ninja with CMAKE_BUILD_TYPE=Debug, Release, "
            "RelWithDebInfo, or MinSizeRel")
    endif()

    # Normalize user input for comparisons and CMAKE_<LANG>_FLAGS_<CONFIG> lookups.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)

    if(VORTEX_CARGO_PROFILE)
        # An explicit user override takes precedence over the CMake build-type mapping.
        _vortex_reject_semicolon("VORTEX_CARGO_PROFILE" "${VORTEX_CARGO_PROFILE}")
        if(NOT VORTEX_CARGO_PROFILE MATCHES "^[A-Za-z0-9_-]+$")
            message(FATAL_ERROR
                "VORTEX_CARGO_PROFILE must contain only letters, numbers, "
                "underscores, or hyphens; got '${VORTEX_CARGO_PROFILE}'")
        endif()
        if(VORTEX_CARGO_PROFILE STREQUAL "test" OR VORTEX_CARGO_PROFILE STREQUAL "bench")
            message(FATAL_ERROR
                "VORTEX_CARGO_PROFILE=${VORTEX_CARGO_PROFILE} is unsupported because "
                "its artifact location is not stable")
        endif()

        set(_cargo_profile "${VORTEX_CARGO_PROFILE}")
    elseif(_build_type STREQUAL "DEBUG" OR _build_type STREQUAL "")
        # Debug and empty build types map to Cargo's built-in development profile.
        set(_cargo_profile "dev")
    elseif(_build_type STREQUAL "RELEASE")
        # Release maps to Cargo's built-in optimized profile.
        set(_cargo_profile "release")
    elseif(_build_type STREQUAL "RELWITHDEBINFO")
        # RelWithDebInfo keeps release optimizations and full debug information.
        set(_cargo_profile "release_debug")
    elseif(_build_type STREQUAL "MINSIZEREL")
        # MinSizeRel uses the release profile optimized for binary size.
        set(_cargo_profile "release_size")
    else()
        # Unknown build types fall back to Cargo's development profile with a warning.
        message(WARNING
            "Vortex has no Cargo profile mapping for "
            "CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'; using Cargo profile dev. "
            "Set VORTEX_CARGO_PROFILE to override it")
        set(_cargo_profile "dev")
    endif()

    if(_cargo_profile STREQUAL "dev")
        set(_artifact_directory "debug")
    else()
        set(_artifact_directory "${_cargo_profile}")
    endif()

    set(${configuration_output} "${_build_type}" PARENT_SCOPE)
    set(${profile_output} "${_cargo_profile}" PARENT_SCOPE)
    set(${artifact_directory_output} "${_artifact_directory}" PARENT_SCOPE)
endfunction()

# Select the FFI package, archive, headers, and optional CUDA tools.
function(_vortex_resolve_ffi_package
    workspace_root
    package_output
    archive_name_output
    headers_output
    nvcc_output
    cuda_root_output)
    set(_package "vortex-ffi")
    set(_archive_name "libvortex_ffi.a")
    set(_manifest "${workspace_root}/vortex-ffi/Cargo.toml")
    set(_headers "${workspace_root}/vortex-ffi/cinclude/vortex.h")

    # Shadow parent-scope values so CPU-only builds return empty CUDA outputs.
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
        list(APPEND _headers "${workspace_root}/vortex-cuda/ffi/cinclude/vortex_cuda.h")
        # NVCC locates its configuration and includes relative to its invocation path.
        # Resolve toolkit symlinks without changing the CUDA target root.
        file(REAL_PATH "${CUDAToolkit_NVCC_EXECUTABLE}" _nvcc)
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
    set(${headers_output} "${_headers}" PARENT_SCOPE)
    set(${nvcc_output} "${_nvcc}" PARENT_SCOPE)
    set(${cuda_root_output} "${_cuda_root}" PARENT_SCOPE)
endfunction()

# Configure sanitizer flags for target code, not host build tools.
# UBSan is native-only; Rust standard-library instrumentation is optional.
function(_vortex_resolve_sanitizer
    configuration
    native_flag_output
    rustflags_output
    build_std_output)
    string(REPLACE "," ";" _sanitizers "${VORTEX_SANITIZER}")
    string(TOLOWER "${_sanitizers}" _sanitizers)
    list(TRANSFORM _sanitizers STRIP)
    list(FILTER _sanitizers EXCLUDE REGEX "^$")

    set(_native "")
    set(_rust "")
    foreach(_sanitizer IN LISTS _sanitizers)
        if(_sanitizer STREQUAL "asan")
            list(APPEND _native address)
            list(APPEND _rust address)
        elseif(_sanitizer STREQUAL "lsan")
            list(APPEND _native leak)
            list(APPEND _rust leak)
        elseif(_sanitizer STREQUAL "ubsan")
            list(APPEND _native undefined)
        elseif(_sanitizer STREQUAL "tsan")
            list(APPEND _native thread)
            list(APPEND _rust thread)
        else()
            message(FATAL_ERROR
                "VORTEX_SANITIZER accepts asan, lsan, ubsan, and tsan; got '${_sanitizer}'")
        endif()
    endforeach()

    # Rebuilding std is meaningful only when Rust itself is instrumented.
    if(VORTEX_SANITIZE_RUST_STD AND NOT _rust)
        message(FATAL_ERROR
            "VORTEX_SANITIZE_RUST_STD=ON requires a Rust sanitizer: asan, lsan, or tsan")
    endif()

    set(_native_flag "")
    set(_rustflags "")
    if(_native)
        # Use Clang so Rust and native code share one compatible sanitizer runtime.
        if(NOT CMAKE_C_COMPILER_ID MATCHES "^(AppleClang|Clang)$" OR
            NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
            message(FATAL_ERROR
                "Vortex sanitizer builds require Clang or AppleClang for C and C++; "
                "found ${CMAKE_C_COMPILER_ID} and ${CMAKE_CXX_COMPILER_ID}")
        endif()
        if(NOT configuration STREQUAL "DEBUG")
            message(FATAL_ERROR "Vortex sanitizer builds require CMAKE_BUILD_TYPE=Debug")
        endif()

        list(JOIN _native "," _native_joined)
        set(_native_flag "-fsanitize=${_native_joined}")
    endif()
    if(_rust)
        # Rust's instrumentation expects upstream compiler-rt, not Apple's runtime.
        if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang" OR
            NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR
                "Rust sanitizer builds require upstream LLVM Clang for C and C++; "
                "set CMAKE_C_COMPILER and CMAKE_CXX_COMPILER to its clang and clang++. "
                "AppleClang is supported only for ubsan")
        endif()
        list(JOIN _rust "," _rust_joined)
        list(APPEND _rustflags
            -A warnings
            -Cunsafe-allow-abi-mismatch=sanitizer
            -C opt-level=0
            # Use the sanitizer runtime linked by the final C++ target for both languages.
            -Zexternal-clangrt
            "-Zsanitizer=${_rust_joined}")
    endif()

    set(${native_flag_output} "${_native_flag}" PARENT_SCOPE)
    set(${rustflags_output} "${_rustflags}" PARENT_SCOPE)
    set(${build_std_output} "${VORTEX_SANITIZE_RUST_STD}" PARENT_SCOPE)
endfunction()

# Reconstruct CMake's effective C and C++ flags for Cargo build scripts, including
# Apple SDK, deployment-target, sanitizer, and PIC requirements.
#
# These flags reach build.rs through Cargo's environment, where the cc crate uses
# them to build native dependencies consistently with the parent CMake project.
function(_vortex_native_flags
    configuration
    apple_deployment_target
    sanitizer_flag
    cflags_output
    cxxflags_output)
    set(_cmake_c_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${configuration}}")
    set(_cmake_cxx_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${configuration}}")
    _vortex_reject_semicolon("effective CMAKE_C_FLAGS" "${_cmake_c_flags}")
    _vortex_reject_semicolon("effective CMAKE_CXX_FLAGS" "${_cmake_cxx_flags}")
    separate_arguments(_cflags UNIX_COMMAND "${_cmake_c_flags}")
    separate_arguments(_cxxflags UNIX_COMMAND "${_cmake_cxx_flags}")

    # Cargo's vendored dependencies do not share the parent's warning policy.
    # Preserve other flags: even non-linker options can be toolchain requirements.
    list(FILTER _cflags EXCLUDE REGEX "^(-Werror(=.*)?|-pedantic-errors)$")
    list(FILTER _cxxflags EXCLUDE REGEX "^(-Werror(=.*)?|-pedantic-errors)$")

    if(APPLE AND CMAKE_OSX_SYSROOT)
        # CMake resolves named SDKs to paths for single-config generators.
        _vortex_reject_semicolon("CMAKE_OSX_SYSROOT" "${CMAKE_OSX_SYSROOT}")
        list(APPEND _cflags -isysroot "${CMAKE_OSX_SYSROOT}")
        list(APPEND _cxxflags -isysroot "${CMAKE_OSX_SYSROOT}")
    endif()

    if(apple_deployment_target)
        # Match Cargo-built native code to CMake's minimum macOS version.
        list(APPEND _cflags "-mmacosx-version-min=${apple_deployment_target}")
        list(APPEND _cxxflags "-mmacosx-version-min=${apple_deployment_target}")
    endif()

    if(sanitizer_flag)
        # Instrument Cargo-built target native code with the selected sanitizer.
        list(APPEND _cflags "${sanitizer_flag}")
        list(APPEND _cxxflags "${sanitizer_flag}")
    endif()

    # Native dependencies become part of the archive embedded in shared parents.
    # Match CMake targets even when cc-rs or parent flags enable debug info.
    list(APPEND _cflags -fPIC -g${VORTEX_DEBUG_INFO})
    list(APPEND _cxxflags -fPIC -g${VORTEX_DEBUG_INFO})

    set(${cflags_output} "${_cflags}" PARENT_SCOPE)
    set(${cxxflags_output} "${_cxxflags}" PARENT_SCOPE)
endfunction()

# Configure Cargo and expose the staged FFI archive as a private dependency of
# the public Vortex C++ target.
block(SCOPE_FOR VARIABLES)
    _vortex_resolve_cargo_profile(_configuration _cargo_profile _cargo_artifact_directory)

    set(_cuda_arch_flags "")
    set(_cuda_host_compiler "")
    if(VORTEX_ENABLE_CUDA)
        _vortex_resolve_cuda_architectures(_cuda_arch_flags)
        _vortex_resolve_cuda_host_compiler(_cuda_host_compiler)
    endif()

    get_filename_component(_workspace_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    _vortex_resolve_ffi_package(
        "${_workspace_root}"
        _ffi_package
        _cargo_archive_name
        _ffi_headers
        _nvcc_executable
        _cuda_root)

    _vortex_resolve_sanitizer(
        "${_configuration}"
        _sanitizer_compile_flag
        _sanitizer_rustflags
        _cargo_build_std)
    _vortex_resolve_rust_toolchain("${_workspace_root}" "${_sanitizer_rustflags}")

    message(STATUS "Vortex supplies Rust flags instead of environment or Cargo configuration rustflags")

    _vortex_native_flags(
        "${_configuration}"
        "${VORTEX_APPLE_DEPLOYMENT_TARGET}"
        "${_sanitizer_compile_flag}"
        _native_c_flags
        _native_cxx_flags)
    # Mirror .cargo/config.toml's Unix rustflags, which CARGO_ENCODED_RUSTFLAGS overrides.
    # Keep them in sync; PIC additionally allows embedding in shared libraries.
    set(_rustflags ${_sanitizer_rustflags}
        -C force-frame-pointers=yes
        -C relocation-model=pic
        -C "debuginfo=${VORTEX_DEBUG_INFO}")

    # Cargo owns incremental invalidation inside this CMake-build-local cache.
    # Registering the directory as additional clean state gives the standard
    # CMake clean target the same effect as `cargo clean --target-dir ...`.
    set(_cargo_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${_cargo_target_dir}")
    set(_cargo_ffi_archive
        "${_cargo_target_dir}/${VORTEX_RUST_TARGET}/${_cargo_artifact_directory}/${_cargo_archive_name}")
    set(_ffi_archive "${CMAKE_CURRENT_BINARY_DIR}/vortex-artifacts/libvortex_ffi.a")
    set(_ffi_include_dir "${CMAKE_CURRENT_BINARY_DIR}/vortex-artifacts/include")
    # Imported include directories must exist before the first build.
    file(MAKE_DIRECTORY "${_ffi_include_dir}")
    # Clean also owns headers staged by previous configurations of this build tree.
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${_ffi_include_dir}")
    if(NOT VORTEX_ENABLE_CUDA)
        # A reused CUDA build tree must not expose the CUDA API in CPU-only mode.
        file(REMOVE "${_ffi_include_dir}/vortex_cuda.h")
    endif()
    set(_staged_headers "")
    foreach(_header IN LISTS _ffi_headers)
        get_filename_component(_header_name "${_header}" NAME)
        list(APPEND _staged_headers "${_ffi_include_dir}/${_header_name}")
    endforeach()

    # Each value below becomes one `-D` argument of the driver and later one
    # environment entry, where a semicolon would split it. The flag and header
    # variables are passed as lists on purpose.
    foreach(_name IN ITEMS
        VORTEX_CARGO_EXECUTABLE VORTEX_RUSTC_EXECUTABLE VORTEX_RUSTUP_TOOLCHAIN
        VORTEX_APPLE_DEPLOYMENT_TARGET
        CMAKE_C_COMPILER CMAKE_CXX_COMPILER
        CMAKE_C_COMPILER_ARG1 CMAKE_CXX_COMPILER_ARG1 CMAKE_AR CMAKE_RANLIB)
        _vortex_reject_semicolon("${_name}" "${${_name}}")
    endforeach()

    # Standalone FFI builds have no compiled CMake target to pull in the archive.
    # Embedded builds run Cargo only when a consumer needs it.
    set(_cargo_default_target "")
    if(PROJECT_IS_TOP_LEVEL)
        set(_cargo_default_target ALL)
    endif()

    # The phony target lets Cargo own dependency tracking. Copy-if-different in
    # the driver prevents fresh Cargo checks from forcing downstream rebuilds.
    # Declare staged headers so Ninja reevaluates consumers after Cargo runs.
    # Source headers are not byproducts: clean must never delete checkout files.
    add_custom_target(vortex_ffi_cargo_build ${_cargo_default_target}
        COMMAND "${CMAKE_COMMAND}"
            "-DVORTEX_CARGO_EXECUTABLE=${VORTEX_CARGO_EXECUTABLE}"
            "-DVORTEX_RUSTC_EXECUTABLE=${VORTEX_RUSTC_EXECUTABLE}"
            "-DVORTEX_RUSTUP_TOOLCHAIN=${VORTEX_RUSTUP_TOOLCHAIN}"
            "-DVORTEX_RUST_TARGET=${VORTEX_RUST_TARGET}"
            "-DVORTEX_CARGO_TARGET_DIR=${_cargo_target_dir}"
            "-DVORTEX_CARGO_PROFILE=${_cargo_profile}"
            "-DVORTEX_FFI_PACKAGE=${_ffi_package}"
            "-DVORTEX_CARGO_FFI_ARCHIVE=${_cargo_ffi_archive}"
            "-DVORTEX_NVCC_EXECUTABLE=${_nvcc_executable}"
            "-DVORTEX_CUDA_ROOT=${_cuda_root}"
            "-DVORTEX_CUDA_ARCH_FLAGS=${_cuda_arch_flags}"
            "-DVORTEX_CUDA_HOST_COMPILER=${_cuda_host_compiler}"
            "-DVORTEX_CARGO_BUILD_STD=${_cargo_build_std}"
            "-DVORTEX_CMAKE_FFI_ARCHIVE=${_ffi_archive}"
            "-DVORTEX_FFI_HEADERS=${_ffi_headers}"
            "-DVORTEX_CMAKE_FFI_INCLUDE_DIR=${_ffi_include_dir}"
            "-DVORTEX_RUSTFLAGS=${_rustflags}"
            "-DVORTEX_CFLAGS=${_native_c_flags}"
            "-DVORTEX_CXXFLAGS=${_native_cxx_flags}"
            "-DVORTEX_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DVORTEX_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DVORTEX_C_COMPILER_ARG1=${CMAKE_C_COMPILER_ARG1}"
            "-DVORTEX_CXX_COMPILER_ARG1=${CMAKE_CXX_COMPILER_ARG1}"
            "-DVORTEX_AR=${CMAKE_AR}"
            "-DVORTEX_RANLIB=${CMAKE_RANLIB}"
            "-DVORTEX_APPLE_DEPLOYMENT_TARGET=${VORTEX_APPLE_DEPLOYMENT_TARGET}"
            -P "${CMAKE_CURRENT_LIST_DIR}/CargoBuild.cmake"
        BYPRODUCTS "${_ffi_archive}" ${_staged_headers}
        COMMENT "Building the PIC Vortex FFI static archive with Cargo"
        USES_TERMINAL
        VERBATIM)

    # Global so that sibling directories such as lang/cpp can link it.
    add_library(vortex_ffi_static STATIC IMPORTED GLOBAL)
    set_target_properties(vortex_ffi_static PROPERTIES
        IMPORTED_LOCATION "${_ffi_archive}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ffi_include_dir}")
    add_dependencies(vortex_ffi_static vortex_ffi_cargo_build)
    _vortex_attach_system_dependencies(vortex_ffi_static "${VORTEX_RUST_TARGET}")
    # Link Rust once, and keep its internal symbols out of the shared C ABI.
    add_library(vortex_ffi_shared SHARED "${CMAKE_CURRENT_LIST_DIR}/shared.c")
    if(NOT BUILD_SHARED_LIBS OR NOT PROJECT_IS_TOP_LEVEL)
        set_target_properties(vortex_ffi_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
    target_link_libraries(vortex_ffi_shared PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,vortex_ffi_static>")
    target_include_directories(vortex_ffi_shared INTERFACE "${_ffi_include_dir}")
    set_target_properties(vortex_ffi_shared PROPERTIES OUTPUT_NAME vortex_ffi)
    if(APPLE)
        target_link_options(vortex_ffi_shared PRIVATE
            # Export only the C ABI; Mach-O prefixes C symbols with an underscore.
            "LINKER:-exported_symbol,_vx_*"
            # Discard unreachable code and data pulled in by whole-archive linking.
            "LINKER:-dead_strip")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Keep the generated linker script in the build tree.
        set(_exports "${CMAKE_CURRENT_BINARY_DIR}/vortex-ffi-exports.map")
        # Export vx_*; localize all other symbols. GENERATE preserves unchanged timestamps.
        file(GENERATE OUTPUT "${_exports}" CONTENT "{ global: vx_*; local: *; };\n")
        target_link_options(vortex_ffi_shared PRIVATE
            # Apply the export allowlist to the Rust archive and its dependencies.
            "LINKER:--version-script=${_exports}"
            # Discard sections unreachable from exported symbols and other linker roots.
            "LINKER:--gc-sections")
        # Relink if the generated export policy changes.
        set_property(TARGET vortex_ffi_shared APPEND PROPERTY LINK_DEPENDS "${_exports}")

        # GNU ld depfiles break on spaces; CMake already tracks the archive and script.
        # Defer disabling linker depfiles so child directories keep their settings.
        cmake_language(DEFER CALL set CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
    else()
        message(FATAL_ERROR "Vortex shared-library exports are not configured for ${CMAKE_SYSTEM_NAME}")
    endif()
    if(_sanitizer_compile_flag)
        foreach(_target IN ITEMS vortex_ffi_static vortex_ffi_shared)
            target_compile_options(${_target} INTERFACE "${_sanitizer_compile_flag}")
            target_link_options(${_target} INTERFACE "${_sanitizer_compile_flag}")
        endforeach()
    endif()

    message(STATUS "Vortex Rust target: ${VORTEX_RUST_TARGET}")
    message(STATUS "Vortex Cargo target directory: ${_cargo_target_dir}")
endblock()
