# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Internal quoting and file-generation primitives shared by the CMake modules.
# Value-producing helpers assign named outputs with PARENT_SCOPE because CMake
# functions otherwise isolate their variables.

include_guard(GLOBAL)

# Fail if a scalar value contains CMake's semicolon list separator.
function(_vortex_reject_semicolon label value)
    if("${value}" MATCHES ";")
        message(FATAL_ERROR
            "${label} contains a semicolon, which is unsupported because "
            "CMake uses semicolons as list separators: ${value}")
    endif()
endfunction()

# Write content only when path is missing or has different contents.
function(_vortex_write_if_different path content)
    if(EXISTS "${path}")
        file(READ "${path}" _existing_content)
        if("${_existing_content}" STREQUAL "${content}")
            return()
        endif()
    endif()
    file(WRITE "${path}" "${content}")
endfunction()

# Encode ARGN as POSIX shell words in one space-delimited string.
function(_vortex_encode_shell_arguments output)
    set(_encoded "")
    foreach(_argument IN LISTS ARGN)
        string(REPLACE "'" "'\"'\"'" _argument_quoted "${_argument}")
        if(_encoded)
            string(APPEND _encoded " ")
        endif()
        string(APPEND _encoded "'${_argument_quoted}'")
    endforeach()
    set(${output} "${_encoded}" PARENT_SCOPE)
endfunction()

# Return the compiler path directly, or create a POSIX wrapper that preserves
# `arg1`. The wrapper filename hashes the complete command so Cargo observes a
# changed CC/CXX value when CMake changes the otherwise-hidden prefix argument.
function(_vortex_make_compiler_wrapper compiler arg1 directory name output)
    if(arg1 STREQUAL "")
        set(${output} "${compiler}" PARENT_SCOPE)
        return()
    endif()

    separate_arguments(_compiler_arg1 NATIVE_COMMAND "${arg1}")
    _vortex_encode_shell_arguments(
        _compiler_command "${compiler}" ${_compiler_arg1})
    set(_script "#!/bin/sh\nexec ${_compiler_command} \"$@\"\n")
    string(SHA256 _command_hash "${_script}")
    string(SUBSTRING "${_command_hash}" 0 16 _command_hash)

    set(_wrapper "${directory}/${name}-${_command_hash}")
    _vortex_write_if_different("${_wrapper}" "${_script}")
    file(CHMOD "${_wrapper}"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE)
    set(${output} "${_wrapper}" PARENT_SCOPE)
endfunction()
