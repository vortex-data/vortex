# SPDX-License-Identifier: CC-BY-4.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# allow linking with vortex_ffi_shared although it's not in current folder
cmake_policy(SET CMP0079 NEW)

add_executable(scan scan.c)
target_link_libraries(scan PRIVATE vortex_ffi_shared)

add_executable(scan_to_arrow scan_to_arrow.c)
target_link_libraries(scan_to_arrow PRIVATE
    nanoarrow_shared vortex_ffi_shared)

add_executable(dtype dtype.c)
target_link_libraries(dtype PRIVATE vortex_ffi_shared)

add_executable(write_sample write_sample.c)
target_link_libraries(write_sample PRIVATE vortex_ffi_shared)
