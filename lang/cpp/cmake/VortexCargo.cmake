# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Configure-time integration that builds vortex-ffi with Cargo and exposes it as
# a CMake static-library target. Its lifecycle is validation -> toolchain
# resolution -> fingerprinting -> support files -> custom target -> imported
# target.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/VortexHelpers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/VortexRustToolchain.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/VortexStaticLink.cmake")

# Normalize the user-facing comma-separated feature scalar into the canonical
# value written to `output` in PARENT_SCOPE. Semicolons and features outside the
# allowlist are fatal because they cannot participate in a supported build.
function(_vortex_normalize_cargo_features input output)
    _vortex_reject_semicolon("VORTEX_CARGO_FEATURES" "${input}")
    string(REPLACE "," ";" _features "${input}")
    set(_normalized_features)
    foreach(_feature IN LISTS _features)
        string(STRIP "${_feature}" _feature)
        if(_feature STREQUAL "")
            continue()
        endif()
        if(NOT _feature STREQUAL "mimalloc")
            message(FATAL_ERROR
                "Unsupported vortex-ffi Cargo feature '${_feature}'. The "
                "validated feature sets are empty and mimalloc.")
        endif()
        list(APPEND _normalized_features "${_feature}")
    endforeach()
    list(REMOVE_DUPLICATES _normalized_features)
    list(SORT _normalized_features)
    string(JOIN "," _normalized_features ${_normalized_features})
    set(${output} "${_normalized_features}" PARENT_SCOPE)
endfunction()

# Validate the tokenized rustc arguments in ARGN against portability policy.
# This function has no output; target-cpu=native and incomplete codegen options
# are fatal. Recognizing every supported -C/--codegen spelling prevents
# alternate syntax from bypassing the check.
function(_vortex_validate_rustflags)
    set(_expect_codegen_option FALSE)
    foreach(_rustflag IN LISTS ARGN)
        set(_codegen_option "")
        if(_expect_codegen_option)
            set(_codegen_option "${_rustflag}")
            set(_expect_codegen_option FALSE)
        elseif(_rustflag STREQUAL "-C" OR _rustflag STREQUAL "--codegen")
            set(_expect_codegen_option TRUE)
            continue()
        elseif(_rustflag MATCHES "^-C=(.+)$")
            set(_codegen_option "${CMAKE_MATCH_1}")
        elseif(_rustflag MATCHES "^-C(.+)$")
            set(_codegen_option "${CMAKE_MATCH_1}")
        elseif(_rustflag MATCHES "^--codegen=(.+)$")
            set(_codegen_option "${CMAKE_MATCH_1}")
        else()
            continue()
        endif()

        if(_codegen_option MATCHES "^target-cpu=(.+)$")
            string(TOLOWER "${CMAKE_MATCH_1}" _target_cpu)
            if(_target_cpu STREQUAL "native")
                message(FATAL_ERROR
                    "Vortex builds do not support Rust target-cpu=native because "
                    "build-host CPU features are unsafe for distributed parent "
                    "artifacts")
            endif()
        endif()
    endforeach()

    if(_expect_codegen_option)
        message(FATAL_ERROR "VORTEX_RUSTFLAGS ends with an incomplete codegen option")
    endif()
endfunction()

# Build the repository's vortex-ffi crate as a private C++ dependency.
# VORTEX_* options supply build policy. Unsupported configuration or incoherent
# toolchain metadata is fatal.
function(_vortex_add_ffi_static)
    if(ARGC GREATER 0)
        message(FATAL_ERROR "_vortex_add_ffi_static does not accept arguments")
    endif()
    if(TARGET vortex_ffi_static)
        message(FATAL_ERROR "The internal vortex_ffi_static target already exists")
    endif()
    _vortex_normalize_cargo_features("${VORTEX_CARGO_FEATURES}" _cargo_features)

    # One Cargo profile is selected per invocation, so multi-config generators
    # cannot share this target without ambiguous artifact locations.
    if(CMAKE_CONFIGURATION_TYPES)
        message(FATAL_ERROR
            "The initial Vortex CMake integration supports single-config "
            "generators only; use Ninja with CMAKE_BUILD_TYPE=Debug, Release, "
            "or RelWithDebInfo")
    endif()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _configuration)
    if(_configuration STREQUAL "")
        message(STATUS
            "Vortex: CMAKE_BUILD_TYPE is empty; using the Debug Cargo "
            "profile without modifying the parent build type")
    elseif(NOT _configuration STREQUAL "DEBUG" AND
        NOT _configuration STREQUAL "RELEASE" AND
        NOT _configuration STREQUAL "RELWITHDEBINFO")
        message(FATAL_ERROR
            "The ${CMAKE_BUILD_TYPE} CMake configuration has no configured "
            "Cargo profile. Supported configurations are Debug, Release, and "
            "RelWithDebInfo.")
    endif()

    # Keep sanitizer builds on their validated nightly/Debug path and reject the
    # allocator combination whose behavior cannot be guaranteed.
    string(TOLOWER "${VORTEX_SANITIZER}" _sanitizer)
    if(NOT _sanitizer STREQUAL "" AND
        NOT _sanitizer STREQUAL "asan" AND
        NOT _sanitizer STREQUAL "tsan")
        message(FATAL_ERROR
            "VORTEX_SANITIZER must be empty, asan, or tsan; got "
            "${VORTEX_SANITIZER}")
    endif()
    if(NOT _sanitizer STREQUAL "" AND _cargo_features STREQUAL "mimalloc")
        message(FATAL_ERROR
            "VORTEX_CARGO_FEATURES=mimalloc is incompatible with "
            "VORTEX_SANITIZER=${_sanitizer}; disable mimalloc for sanitizer "
            "builds")
    endif()
    if(NOT _sanitizer STREQUAL "" AND
        NOT _configuration STREQUAL "" AND
        NOT _configuration STREQUAL "DEBUG")
        message(FATAL_ERROR "Vortex sanitizer builds require CMAKE_BUILD_TYPE=Debug")
    endif()

    if(VORTEX_CARGO_JOBS AND NOT VORTEX_CARGO_JOBS MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "VORTEX_CARGO_JOBS must be a positive integer")
    endif()

    # Keep the CMake configuration, Cargo profile, and Cargo artifact directory
    # in a single explicit mapping used by both the command and staged path.
    if(_configuration STREQUAL "" OR _configuration STREQUAL "DEBUG")
        set(_configuration_name "Debug")
        set(_cargo_profile "dev")
        set(_cargo_artifact_directory "debug")
    elseif(_configuration STREQUAL "RELEASE")
        set(_configuration_name "Release")
        set(_cargo_profile "release")
        set(_cargo_artifact_directory "release")
    else()
        set(_configuration_name "RelWithDebInfo")
        set(_cargo_profile "release_debug")
        set(_cargo_artifact_directory "release_debug")
    endif()

    # Canonicalize workspace identity before resolving tools. Values that cross
    # CMake list and COMMAND boundaries must remain scalar: a semicolon would
    # split one path or shell-style flag string into multiple elements.
    get_filename_component(_workspace_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../.." ABSOLUTE)
    set(_ffi_source_dir "${_workspace_root}/vortex-ffi")
    set(_ffi_manifest "${_ffi_source_dir}/Cargo.toml")
    if(NOT EXISTS "${_workspace_root}/Cargo.toml" OR
        NOT EXISTS "${_workspace_root}/Cargo.lock" OR
        NOT EXISTS "${_ffi_manifest}")
        message(FATAL_ERROR
            "Vortex's CMake source build requires a complete workspace "
            "checkout with Cargo.toml, Cargo.lock, and vortex-ffi/Cargo.toml")
    endif()
    file(REAL_PATH "${_workspace_root}" _workspace_real_path)
    _vortex_reject_semicolon("Vortex workspace path" "${_workspace_root}")
    _vortex_reject_semicolon("Vortex workspace real path" "${_workspace_real_path}")
    _vortex_reject_semicolon("Vortex FFI source path" "${_ffi_source_dir}")
    _vortex_reject_semicolon("Vortex FFI manifest path" "${_ffi_manifest}")
    _vortex_reject_semicolon("Vortex binary path" "${CMAKE_CURRENT_BINARY_DIR}")
    _vortex_reject_semicolon("VORTEX_CARGO_TARGET_DIR" "${VORTEX_CARGO_TARGET_DIR}")
    _vortex_reject_semicolon("VORTEX_RUSTFLAGS" "${VORTEX_RUSTFLAGS}")

    # Resolve rustup proxies to concrete Cargo/rustc binaries, then verify the
    # Rust target and native C/C++ compilers describe the same host ABI.
    _vortex_find_rust_tools("${_workspace_root}")
    if(NOT _sanitizer STREQUAL "" AND NOT VORTEX_RESOLVED_RUSTC_RELEASE MATCHES "nightly")
        message(FATAL_ERROR
            "VORTEX_SANITIZER=${_sanitizer} requires a nightly Rust "
            "toolchain; found ${VORTEX_RESOLVED_RUSTC_RELEASE}")
    endif()
    _vortex_resolve_native_target("${VORTEX_RESOLVED_RUSTC_EXECUTABLE}" "${_workspace_root}" _rust_target)
    _vortex_resolve_apple_deployment_target(_apple_deployment_target)
    _vortex_resolve_apple_sdkroot(_apple_sdkroot _apple_sdk_version)

    string(TOUPPER "${_rust_target}" _target_env_upper)
    string(REPLACE "-" "_" _target_env_upper "${_target_env_upper}")
    string(TOLOWER "${_rust_target}" _target_env_lower)
    string(REPLACE "-" "_" _target_env_lower "${_target_env_lower}")

    # Parse the opted-in flag string with shell rules so rustc receives the
    # intended argv. Ambient Rust flags are excluded because unvalidated state
    # must not evade portability checks or the build fingerprint.
    set(_user_rustflags)
    if(VORTEX_RUSTFLAGS)
        separate_arguments(_user_rustflags UNIX_COMMAND "${VORTEX_RUSTFLAGS}")
    endif()
    set(_ambient_rustflags FALSE)
    if(DEFINED ENV{CARGO_ENCODED_RUSTFLAGS})
        if(NOT "$ENV{CARGO_ENCODED_RUSTFLAGS}" STREQUAL "")
            set(_ambient_rustflags TRUE)
        endif()
    endif()
    if(DEFINED ENV{RUSTFLAGS})
        if(NOT "$ENV{RUSTFLAGS}" STREQUAL "")
            set(_ambient_rustflags TRUE)
        endif()
    endif()
    if(_ambient_rustflags)
        message(STATUS
            "Vortex ignores ambient Rust flags; use VORTEX_RUSTFLAGS so "
            "the validated flags participate in the CMake fingerprint")
    endif()
    set(_cargo_build_std OFF)
    set(_cargo_no_default_features ON)
    set(_sanitizer_compile_flag "")
    if(_sanitizer STREQUAL "asan")
        set(_cargo_build_std ON)
        set(_sanitizer_compile_flag "-fsanitize=address,undefined,leak")
        list(APPEND _user_rustflags
            -A warnings
            -Cunsafe-allow-abi-mismatch=sanitizer
            -C debuginfo=2
            -C opt-level=0
            -C strip=none
            -Zexternal-clangrt
            -Zsanitizer=address,leak)
    elseif(_sanitizer STREQUAL "tsan")
        set(_cargo_build_std ON)
        set(_sanitizer_compile_flag "-fsanitize=thread")
        list(APPEND _user_rustflags
            -A warnings
            -Cunsafe-allow-abi-mismatch=sanitizer
            -C debuginfo=2
            -C opt-level=0
            -C strip=none
            -Zexternal-clangrt
            -Zsanitizer=thread)
    endif()

    _vortex_validate_rustflags(${_user_rustflags})

    # Preserve CMake's separate compile and link sysroots. C/C++ compilations from
    # Cargo build scripts consume the former, while rustc forwards the latter at
    # link.
    if(CMAKE_SYSROOT_COMPILE)
        set(_compile_sysroot "${CMAKE_SYSROOT_COMPILE}")
    elseif(CMAKE_SYSROOT)
        set(_compile_sysroot "${CMAKE_SYSROOT}")
    elseif(APPLE)
        set(_compile_sysroot "${_apple_sdkroot}")
    else()
        set(_compile_sysroot "")
    endif()
    if(CMAKE_SYSROOT_LINK)
        set(_link_sysroot "${CMAKE_SYSROOT_LINK}")
    elseif(CMAKE_SYSROOT)
        set(_link_sysroot "${CMAKE_SYSROOT}")
    elseif(APPLE)
        set(_link_sysroot "${_apple_sdkroot}")
    else()
        set(_link_sysroot "")
    endif()

    # Cargo build scripts must use the exact CMake-selected native toolchain.
    # Compiler wrappers generated below retain any CMAKE_*_COMPILER_ARG1 prefix.
    _vortex_optional_value(CMAKE_C_COMPILER_ARG1 _c_compiler_arg1)
    _vortex_optional_value(CMAKE_C_COMPILER_TARGET _c_compiler_target)
    _vortex_optional_value(CMAKE_CXX_COMPILER _cxx_compiler)
    _vortex_optional_value(CMAKE_CXX_COMPILER_ARG1 _cxx_compiler_arg1)
    _vortex_optional_value(CMAKE_CXX_COMPILER_ID _cxx_compiler_id)
    _vortex_optional_value(CMAKE_CXX_COMPILER_VERSION _cxx_compiler_version)
    _vortex_optional_value(CMAKE_CXX_COMPILER_TARGET _cxx_compiler_target)
    _vortex_optional_value(CMAKE_AR _archiver)
    _vortex_optional_value(CMAKE_RANLIB _ranlib)

    if(_c_compiler_target AND CMAKE_C_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        set(_native_c_compiler_target "${_c_compiler_target}")
    else()
        set(_native_c_compiler_target "")
    endif()
    if(_cxx_compiler_target AND _cxx_compiler_id MATCHES "^(AppleClang|Clang)$")
        set(_native_cxx_compiler_target "${_cxx_compiler_target}")
    else()
        set(_native_cxx_compiler_target "")
    endif()

    _vortex_reject_semicolon("CMAKE_C_COMPILER" "${CMAKE_C_COMPILER}")
    _vortex_reject_semicolon("CMAKE_C_COMPILER_ARG1" "${_c_compiler_arg1}")
    _vortex_reject_semicolon("CMAKE_CXX_COMPILER" "${_cxx_compiler}")
    _vortex_reject_semicolon("CMAKE_CXX_COMPILER_ARG1" "${_cxx_compiler_arg1}")
    _vortex_reject_semicolon("CMAKE_AR" "${_archiver}")
    _vortex_reject_semicolon("CMAKE_RANLIB" "${_ranlib}")
    _vortex_reject_semicolon("compile sysroot" "${_compile_sysroot}")
    _vortex_reject_semicolon("link sysroot" "${_link_sysroot}")

    # Reconstruct effective configuration-specific native flags as argv, then add
    # target, sysroot, deployment, sanitizer, and PIC requirements consistently
    # for native dependencies compiled from Cargo build scripts.
    _vortex_optional_value(CMAKE_C_FLAGS _cmake_c_flags)
    _vortex_optional_value(CMAKE_CXX_FLAGS _cmake_cxx_flags)
    if(NOT _configuration STREQUAL "")
        _vortex_optional_value(CMAKE_C_FLAGS_${_configuration} _cmake_c_configuration_flags)
        _vortex_optional_value(CMAKE_CXX_FLAGS_${_configuration} _cmake_cxx_configuration_flags)
        string(APPEND _cmake_c_flags " ${_cmake_c_configuration_flags}")
        string(APPEND _cmake_cxx_flags " ${_cmake_cxx_configuration_flags}")
    endif()
    _vortex_reject_semicolon("effective CMAKE_C_FLAGS" "${_cmake_c_flags}")
    _vortex_reject_semicolon("effective CMAKE_CXX_FLAGS" "${_cmake_cxx_flags}")
    separate_arguments(_native_c_flags UNIX_COMMAND "${_cmake_c_flags}")
    separate_arguments(_native_cxx_flags UNIX_COMMAND "${_cmake_cxx_flags}")
    if(_compile_sysroot)
        list(APPEND _native_c_flags "--sysroot=${_compile_sysroot}")
        list(APPEND _native_cxx_flags "--sysroot=${_compile_sysroot}")
    endif()
    if(_native_c_compiler_target)
        list(APPEND _native_c_flags "--target=${_native_c_compiler_target}")
    endif()
    if(_native_cxx_compiler_target)
        list(APPEND _native_cxx_flags "--target=${_native_cxx_compiler_target}")
    elseif(_native_c_compiler_target AND NOT _cxx_compiler)
        list(APPEND _native_cxx_flags "--target=${_native_c_compiler_target}")
    endif()
    if(_apple_deployment_target)
        list(APPEND _native_c_flags "-mmacosx-version-min=${_apple_deployment_target}")
        list(APPEND _native_cxx_flags "-mmacosx-version-min=${_apple_deployment_target}")
    endif()
    if(_sanitizer_compile_flag)
        list(APPEND _native_c_flags "${_sanitizer_compile_flag}")
        list(APPEND _native_cxx_flags "${_sanitizer_compile_flag}")
    endif()
    # Native dependencies compiled by Cargo build scripts are part of the static
    # archive and must remain suitable for embedding in a shared parent.
    list(APPEND _native_c_flags -fPIC)
    list(APPEND _native_cxx_flags -fPIC)
    _vortex_encode_shell_arguments(_native_c_flags_shell ${_native_c_flags})
    _vortex_encode_shell_arguments(_native_cxx_flags_shell ${_native_cxx_flags})

    # Cargo's encoded rustflags format uses ASCII unit separator (31) between
    # arguments, preserving exact rustc argv without another shell/list parse.
    set(_final_rustflags ${_user_rustflags})
    list(APPEND _final_rustflags
        -C force-frame-pointers=yes
        -C relocation-model=pic)
    if(_link_sysroot)
        list(APPEND _final_rustflags -C "link-arg=--sysroot=${_link_sysroot}")
    endif()
    if(_native_c_compiler_target)
        list(APPEND _final_rustflags -C "link-arg=--target=${_native_c_compiler_target}")
    endif()
    string(ASCII 31 _rustflags_separator)
    string(JOIN "${_rustflags_separator}" _encoded_rustflags ${_final_rustflags})

    # Hash every resolved toolchain and ABI/codegen input that can affect the
    # archive. A fingerprint-specific Cargo tree isolates incompatible caches;
    # profile outputs are separated again by configuration below.
    set(_fingerprint_input
        "${_workspace_real_path}|"
        "${VORTEX_RESOLVED_CARGO_EXECUTABLE}|${VORTEX_RESOLVED_CARGO_VERBOSE}|"
        "${VORTEX_RESOLVED_RUSTC_EXECUTABLE}|${VORTEX_RESOLVED_RUSTC_VERBOSE}|"
        "${_rust_target}|${_cargo_features}|${_cargo_no_default_features}|"
        "${CMAKE_C_COMPILER}|${_c_compiler_arg1}|"
        "${CMAKE_C_COMPILER_ID}|${CMAKE_C_COMPILER_VERSION}|"
        "${_c_compiler_target}|"
        "${_cxx_compiler}|${_cxx_compiler_arg1}|"
        "${_cxx_compiler_id}|${_cxx_compiler_version}|"
        "${_cxx_compiler_target}|${_archiver}|${_ranlib}|"
        "${_compile_sysroot}|${_link_sysroot}|${_apple_sdkroot}|"
        "${_apple_sdk_version}|"
        "${_apple_deployment_target}|"
        "${_sanitizer}|${_native_c_flags_shell}|"
        "${_native_cxx_flags_shell}|${_encoded_rustflags}")
    string(SHA256 _fingerprint "${_fingerprint_input}")

    if(VORTEX_CARGO_TARGET_DIR)
        get_filename_component(_cargo_target_root
            "${VORTEX_CARGO_TARGET_DIR}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    else()
        set(_cargo_target_root "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
    endif()
    set(_cargo_target_base "${_cargo_target_root}/${_fingerprint}")
    set(_cargo_support_dir "${_cargo_target_base}/cmake-support")

    # An explicit target root may be shared by concurrent configure processes.
    # Hold a function-scoped lock while producing wrappers and support files so a
    # build cannot observe a mixed or partially generated configuration.
    file(MAKE_DIRECTORY "${_cargo_support_dir}")
    file(LOCK "${_cargo_support_dir}/configure.lock"
        GUARD FUNCTION
        TIMEOUT 60
        RESULT_VARIABLE _cargo_support_lock_result)
    if(NOT _cargo_support_lock_result EQUAL 0)
        message(FATAL_ERROR
            "Could not lock Vortex Cargo support directory "
            "${_cargo_support_dir}: ${_cargo_support_lock_result}")
    endif()

    _vortex_make_compiler_wrapper(
        "${CMAKE_C_COMPILER}" "${_c_compiler_arg1}"
        "${_cargo_support_dir}" cc _native_c_compiler)
    if(_cxx_compiler)
        _vortex_make_compiler_wrapper(
            "${_cxx_compiler}" "${_cxx_compiler_arg1}"
            "${_cargo_support_dir}" cxx _native_cxx_compiler)
    else()
        set(_native_cxx_compiler "${_native_c_compiler}")
    endif()

    # Passing this generated file via Cargo --config makes its linker and profile
    # policy explicit rather than dependent on directory discovery. The driver's
    # target-specific environment has higher precedence for linker and rustflags.
    set(_cargo_config "${_cargo_support_dir}/config.toml")
    _vortex_toml_string("${_native_c_compiler}" _linker_toml)
    string(CONCAT _cargo_config_contents
        "[target.\"${_rust_target}\"]\n"
        "linker = ${_linker_toml}\n"
        "\n"
        "[profile.dev]\n"
        "strip = \"none\"\n"
        "\n"
        "[profile.release]\n"
        "codegen-units = 1\n"
        "lto = \"off\"\n"
        "strip = \"none\"\n"
        "\n"
        "[profile.release_debug]\n"
        "debug = \"full\"\n"
        "strip = \"none\"\n")
    _vortex_write_if_different("${_cargo_config}" "${_cargo_config_contents}")

    # Files carry shell-quoted native flags and ASCII-31 Rust flags safely across
    # the configure/build boundary. Write-if-different avoids needless timestamp
    # churn in the shared support directory.
    set(_rustflags_file "${_cargo_support_dir}/rustflags")
    set(_cflags_file "${_cargo_support_dir}/cflags")
    set(_cxxflags_file "${_cargo_support_dir}/cxxflags")
    _vortex_write_if_different("${_rustflags_file}" "${_encoded_rustflags}")
    _vortex_write_if_different("${_cflags_file}" "${_native_c_flags_shell}")
    _vortex_write_if_different("${_cxxflags_file}" "${_native_cxx_flags_shell}")

    set(_archive_name "libvortex_ffi.a")
    set(_cargo_target_dir "${_cargo_target_base}/${_configuration_name}")
    string(CONCAT _cargo_ffi_archive
        "${_cargo_target_dir}/${_rust_target}/"
        "${_cargo_artifact_directory}/${_archive_name}")
    string(CONCAT _ffi_archive
        "${CMAKE_CURRENT_BINARY_DIR}/vortex-artifacts/"
        "${_configuration_name}/${_archive_name}")

    # This phony target intentionally has no OUTPUT, so Cargo checks freshness on
    # every build. BYPRODUCTS describes the staged archive to generators; each -D
    # assignment is one list element and VERBATIM protects the outer cmake -P
    # driver's argument boundaries.
    add_custom_target(vortex_ffi_cargo_build ALL
        COMMAND "${CMAKE_COMMAND}"
            "-DVORTEX_CARGO_EXECUTABLE=${VORTEX_RESOLVED_CARGO_EXECUTABLE}"
            "-DVORTEX_RUSTC_EXECUTABLE=${VORTEX_RESOLVED_RUSTC_EXECUTABLE}"
            "-DVORTEX_RUST_TARGET=${_rust_target}"
            "-DVORTEX_WORKSPACE_ROOT=${_workspace_root}"
            "-DVORTEX_FFI_MANIFEST=${_ffi_manifest}"
            "-DVORTEX_CARGO_CONFIG_FILE=${_cargo_config}"
            "-DVORTEX_CARGO_TARGET_DIR=${_cargo_target_dir}"
            "-DVORTEX_CARGO_PROFILE=${_cargo_profile}"
            "-DVORTEX_CARGO_JOBS=${VORTEX_CARGO_JOBS}"
            "-DVORTEX_CARGO_OFFLINE=${VORTEX_CARGO_OFFLINE}"
            "-DVORTEX_CARGO_FEATURES=${_cargo_features}"
            "-DVORTEX_CARGO_BUILD_STD=${_cargo_build_std}"
            "-DVORTEX_CARGO_NO_DEFAULT_FEATURES=${_cargo_no_default_features}"
            "-DVORTEX_CARGO_FFI_ARCHIVE=${_cargo_ffi_archive}"
            "-DVORTEX_CMAKE_FFI_ARCHIVE=${_ffi_archive}"
            "-DVORTEX_TARGET_ENV_KEY_UPPER=${_target_env_upper}"
            "-DVORTEX_TARGET_ENV_KEY_LOWER=${_target_env_lower}"
            "-DVORTEX_C_LINKER=${_native_c_compiler}"
            "-DVORTEX_C_COMPILER=${_native_c_compiler}"
            "-DVORTEX_CXX_COMPILER=${_native_cxx_compiler}"
            "-DVORTEX_AR=${_archiver}"
            "-DVORTEX_RANLIB=${_ranlib}"
            "-DVORTEX_RUSTFLAGS_FILE=${_rustflags_file}"
            "-DVORTEX_CFLAGS_FILE=${_cflags_file}"
            "-DVORTEX_CXXFLAGS_FILE=${_cxxflags_file}"
            "-DVORTEX_APPLE_DEPLOYMENT_TARGET=${_apple_deployment_target}"
            "-DVORTEX_APPLE_SDKROOT=${_apple_sdkroot}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BuildVortexCargo.cmake"
        BYPRODUCTS "${_ffi_archive}"
        WORKING_DIRECTORY "${_workspace_root}"
        COMMENT "Building the PIC vortex-ffi static archive with Cargo"
        USES_TERMINAL
        VERBATIM)

    # Keep the imported archive scoped to the C++ source directory. The public
    # C++ target carries its include path, native libraries, and build dependency
    # transitively without exposing a second supported consumer target.
    add_library(vortex_ffi_static STATIC IMPORTED)
    set_target_properties(vortex_ffi_static PROPERTIES
        IMPORTED_LOCATION "${_ffi_archive}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ffi_source_dir}/cinclude")
    add_dependencies(vortex_ffi_static vortex_ffi_cargo_build)
    _vortex_configure_static_link(vortex_ffi_static "${_rust_target}")
    if(_sanitizer_compile_flag)
        target_compile_options(vortex_ffi_static INTERFACE "${_sanitizer_compile_flag}")
        target_link_options(vortex_ffi_static INTERFACE "${_sanitizer_compile_flag}")
    endif()

    message(STATUS "Vortex Rust target: ${_rust_target}")
    message(STATUS "Vortex Cargo target base: ${_cargo_target_base}")
endfunction()
