# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Target names carry a prefix so a build that also includes the C examples has
# no clashes; the executables keep their plain names.
foreach(_name reader writer dtype scan scan_to_arrow)
    add_executable(cpp_${_name} ${_name}.cpp)
    set_target_properties(cpp_${_name} PROPERTIES OUTPUT_NAME ${_name})
    target_link_libraries(cpp_${_name} PRIVATE Vortex::cpp_static nanoarrow_shared)
endforeach()
