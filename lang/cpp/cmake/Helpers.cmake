# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Provides shared internal utilities for the Vortex CMake integration.

include_guard(GLOBAL)

# Fail if a scalar value contains CMake's semicolon list separator.
function(_vortex_reject_semicolon label value)
    if("${value}" MATCHES ";")
        message(FATAL_ERROR
            "${label} contains a semicolon, which is unsupported because "
            "CMake uses semicolons as list separators: ${value}")
    endif()
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
