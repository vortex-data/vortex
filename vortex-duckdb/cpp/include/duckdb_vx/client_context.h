// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_client_context_ *duckdb_vx_client_context;

// Get the client context from a DuckDB connection.
// This reference is valid as long as the connection is valid.
duckdb_vx_client_context duckdb_vx_connection_get_client_context(duckdb_connection conn);

#ifdef __cplusplus /* End C ABI */
}
#endif
