// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stddef.h>

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_error_ *duckdb_vx_error;

//! Create a DuckDB vortex error.
duckdb_vx_error duckdb_vx_error_create(const char *message, size_t message_length);

// Borrows the message owned by the err type.
const char *duckdb_vx_error_value(duckdb_vx_error err);

void duckdb_vx_error_free(duckdb_vx_error err);

#ifdef __cplusplus /* End C ABI */
}
#endif
