# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

foreach(name reader writer dtype scan scan_to_arrow)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE vortex_cxx_shared nanoarrow_shared)
endforeach()
