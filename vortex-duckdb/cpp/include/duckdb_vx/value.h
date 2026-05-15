// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
DUCKDB_INCLUDES_END

#include "duckdb_vx/error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Create a null value with a reference to a logical type.
duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty);

// Unwrap a DuckDB VARIANT value into the typed value it contains.
duckdb_value duckdb_vx_variant_value_unwrap(duckdb_value value,
                                            bool *outer_null,
                                            duckdb_vx_error *err);

#ifdef __cplusplus /* End C ABI */
}
#endif
