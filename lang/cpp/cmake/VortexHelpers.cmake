# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Internal quoting and file-generation primitives shared by the CMake modules.
# Value-producing helpers accept an output variable name and assign it with
# PARENT_SCOPE because CMake functions otherwise isolate their variables.

include_guard(GLOBAL)

# Copy an optional variable to a named output, normalizing undefined to empty.
function(_vortex_optional_value variable output)
    if(DEFINED ${variable})
        set(${output} "${${variable}}" PARENT_SCOPE)
    else()
        set(${output} "" PARENT_SCOPE)
    endif()
endfunction()

# Reject values that cannot remain scalar: CMake stores lists as semicolon-
# delimited strings, so downstream list operations would change argv boundaries.
function(_vortex_reject_semicolon label value)
    if("${value}" MATCHES ";")
        message(FATAL_ERROR
            "${label} contains a semicolon, which is unsupported because "
            "CMake uses semicolons as list separators: ${value}")
    endif()
endfunction()

# Encode a validated scalar as a quoted TOML basic string. These escapes belong
# to TOML and are intentionally distinct from the shell quoting below.
function(_vortex_toml_string input output)
    string(REPLACE "\\" "\\\\" _value "${input}")
    string(REPLACE "\"" "\\\"" _value "${_value}")
    set(${output} "\"${_value}\"" PARENT_SCOPE)
endfunction()

# Encode one POSIX shell word; embedded apostrophes close and reopen the quoted
# segment so the original argument remains a single word.
function(_vortex_shell_quote input output)
    string(REPLACE "'" "'\"'\"'" _quoted "${input}")
    set(${output} "'${_quoted}'" PARENT_SCOPE)
endfunction()

# Preserve generated-file mtimes when content is unchanged, avoiding needless
# downstream CMake or Cargo rebuild triggers.
function(_vortex_write_if_different path content)
    set(_write_file TRUE)
    if(EXISTS "${path}")
        file(READ "${path}" _existing_content)
        if("${_existing_content}" STREQUAL "${content}")
            set(_write_file FALSE)
        endif()
    endif()
    if(_write_file)
        file(WRITE "${path}" "${content}")
    endif()
endfunction()

# Serialize a CMake argv list as a single POSIX-shell command fragment while
# preserving each list element as one shell word.
function(_vortex_encode_shell_arguments output)
    set(_encoded "")
    foreach(_argument IN LISTS ARGN)
        _vortex_shell_quote("${_argument}" _argument_quoted)
        if(_encoded)
            string(APPEND _encoded " ")
        endif()
        string(APPEND _encoded "${_argument_quoted}")
    endforeach()
    set(${output} "${_encoded}" PARENT_SCOPE)
endfunction()

# Return the compiler directly unless CMAKE_<LANG>_COMPILER_ARG1 contributes
# required prefix arguments. Cargo accepts a compiler executable path, so a
# generated wrapper preserves CMake's complete compiler command for CC/CXX.
function(_vortex_make_compiler_wrapper compiler arg1 directory name output)
    if(arg1 STREQUAL "")
        set(${output} "${compiler}" PARENT_SCOPE)
        return()
    endif()

    set(_command "${compiler}")
    separate_arguments(_compiler_arg1 NATIVE_COMMAND "${arg1}")
    list(APPEND _command ${_compiler_arg1})
    set(_script "#!/bin/sh\nexec")
    foreach(_argument IN LISTS _command)
        _vortex_shell_quote("${_argument}" _argument_quoted)
        string(APPEND _script " ${_argument_quoted}")
    endforeach()
    string(APPEND _script " \"$@\"\n")

    set(_wrapper "${directory}/${name}")
    _vortex_write_if_different("${_wrapper}" "${_script}")
    file(CHMOD "${_wrapper}"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE)
    set(${output} "${_wrapper}" PARENT_SCOPE)
endfunction()
