// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Get the client context from a DuckDB connection.
// This reference is valid as long as the connection is valid.
duckdb_client_context duckdb_vx_connection_get_client_context(duckdb_connection conn);

// Try to get the current value of a configuration setting.
// Returns a duckdb_value if the setting exists, or NULL if it doesn't.
// The caller is responsible for freeing the returned value using duckdb_destroy_value.
duckdb_value duckdb_client_context_try_get_current_setting(duckdb_client_context context, const char *key);

#ifdef __cplusplus /* End C ABI */
}
#endif
