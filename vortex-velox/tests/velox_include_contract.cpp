// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

struct ArrowSchema;
struct ArrowArray;
struct ArrowArrayStream;

#define USE_OWN_ARROW
typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#include "vortex_velox.h"
#undef USE_OWN_ARROW

static_assert(VX_VELOX_ABI_VERSION == 1u);
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

    const vx_dtype *(*new_primitive)(vx_velox_ptype, bool, vx_error **) =
        vx_velox_dtype_new_primitive;
    vx_expression *(*new_binary)(vx_velox_binary_operator,
                                 const vx_expression *,
                                 const vx_expression *,
                                 vx_error **) = vx_velox_expression_binary;

    (void)callbacks;
    (void)options;
    (void)new_primitive;
    (void)new_binary;
    (void)vx_velox_source_export_schema;
    (void)vx_velox_data_source_scan;
}
