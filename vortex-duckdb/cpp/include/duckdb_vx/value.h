// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Create a null value with a reference to a logical type.
duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty);

#ifdef __cplusplus /* End C ABI */
}
#endif
