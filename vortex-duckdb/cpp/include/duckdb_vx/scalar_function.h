// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
DUCKDB_INCLUDES_END

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_sfunc_ *duckdb_vx_sfunc;

const char *duckdb_vx_sfunc_name(duckdb_vx_sfunc ffi_func);

duckdb_logical_type duckdb_vx_sfunc_return_type(duckdb_vx_sfunc ffi_func);

#ifdef __cplusplus /* End C ABI */
}
#endif
