# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Prefix targets and keep executables local so a parent's shared output
# directory cannot merge C and C++ examples with the same name.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _config_upper)
foreach(_name reader writer dtype scan scan_to_arrow)
    add_executable(cpp_${_name} ${_name}.cpp)

    set_target_properties(cpp_${_name} PROPERTIES
        OUTPUT_NAME ${_name}
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    # Drop the inherited per-config override so the local output directory takes effect.
    set_property(TARGET cpp_${_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY_${_config_upper})
    target_link_libraries(cpp_${_name} PRIVATE Vortex::cpp)
endforeach()
target_link_libraries(cpp_scan_to_arrow PRIVATE nanoarrow::nanoarrow)
