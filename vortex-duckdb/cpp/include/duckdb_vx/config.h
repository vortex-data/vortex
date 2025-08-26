// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Get a configuration value from a DuckDB config object by key name.
/// Returns a DuckDB value containing the config value, or INVALID if the key doesn't exist.
/// The returned value must be freed with duckdb_destroy_value.
///
/// @param config The DuckDB configuration object
/// @param key The configuration key to retrieve
/// @param out_value Pointer to store the resulting DuckDB value
/// @return DuckDBSuccess on success, DuckDBError if the key doesn't exist or on error
duckdb_state duckdb_vx_get_config_value(duckdb_config config, const char* key, duckdb_value* out_value);

/// Check if a configuration key exists in the given config object.
///
/// @param config The DuckDB configuration object  
/// @param key The configuration key to check
/// @return 1 if the key exists, 0 if it doesn't exist or on error
int duckdb_vx_config_has_key(duckdb_config config, const char* key);

/// Convert a DuckDB value to a string representation.
/// The returned string must be freed with duckdb_free.
///
/// @param value The DuckDB value to convert
/// @return A newly allocated string containing the value's string representation, or NULL on error
char* duckdb_vx_value_to_string(duckdb_value value);

#ifdef __cplusplus
}
#endif