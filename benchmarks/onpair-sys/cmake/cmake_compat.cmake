# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# CMake before 3.24 does not recognize DOWNLOAD_EXTRACT_TIMESTAMP in
# FetchContent_Declare and folds it into URL_HASH. Removing the optional hint
# keeps the pinned, SHA-256-verified Boost download compatible with our minimum
# supported CMake version.
if(NOT DEFINED SRC_DIR)
    message(FATAL_ERROR "cmake_compat.cmake: SRC_DIR not set")
endif()

set(ONPAIR_CMAKE "${SRC_DIR}/CMakeLists.txt")
file(READ "${ONPAIR_CMAKE}" CONTENT)
string(REPLACE
    "        DOWNLOAD_EXTRACT_TIMESTAMP TRUE\n"
    ""
    CONTENT
    "${CONTENT}"
)
file(WRITE "${ONPAIR_CMAKE}" "${CONTENT}")
