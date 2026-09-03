// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

struct ArrowSchema;
struct ArrowArray;
#include "vortex_velox.h"

static_assert(VX_VELOX_ABI_VERSION == 6u);
static_assert(VX_VELOX_SELECTION_ALL == 0);
static_assert(VX_VELOX_OPERATOR_EQ == 0);

void vx_velox_compile_velox_include_contract() {
    vx_velox_read_at_callbacks callbacks {};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.abi_version = VX_VELOX_ABI_VERSION;

    vx_velox_scan_options options {};
    options.struct_size = sizeof(options);
    options.abi_version = VX_VELOX_ABI_VERSION;
    options.selection.include = VX_VELOX_SELECTION_ALL;

    const vx_velox_dtype *(*new_primitive)(vx_velox_ptype, bool, vx_velox_error **) =
        vx_velox_dtype_new_primitive;
    vx_velox_expression *(*new_binary)(vx_velox_binary_operator,
                                       const vx_velox_expression *,
                                       const vx_velox_expression *,
                                       vx_velox_error **) = vx_velox_expression_binary;

    (void)callbacks;
    (void)options;
    (void)new_primitive;
    (void)new_binary;
    (void)vx_velox_source_export_schema;
    (void)vx_velox_data_source_scan;
}
