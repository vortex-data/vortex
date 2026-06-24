// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct duckdb_vx_error_ *duckdb_vx_error;
typedef struct duckdb_vx_data_ *duckdb_vx_data;
typedef struct duckdb_vx_reusable_dict_ *duckdb_vx_reusable_dict;

//! Create a DuckDB vortex error.
duckdb_vx_error duckdb_vx_error_create(const char *message, size_t message_length);

// Borrows the message owned by the err type.
const char *duckdb_vx_error_value(duckdb_vx_error err);

void duckdb_vx_error_free(duckdb_vx_error err);

// Create an opaque data object with a delete callback.
duckdb_vx_data duckdb_vx_data_create(void *data_ptr, duckdb_delete_callback_t delete_callback);

/// Convert a DuckDB value to a string representation.
/// The returned string must be freed with duckdb_free.
char *duckdb_vx_value_to_string(duckdb_value value);

const char *duckdb_data_chunk_to_string(duckdb_data_chunk chunk, duckdb_vx_error *err);

void duckdb_data_chunk_verify(duckdb_data_chunk chunk, duckdb_vx_error *err);

char *duckdb_vx_logical_type_stringify(duckdb_logical_type ty);
duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty);

/// Creates a GEOMETRY logical type with the given CRS (Coordinate Reference System).
/// `crs` must be a NUL-terminated UTF-8 string. Pass an empty string for no CRS.
duckdb_logical_type duckdb_vx_create_geometry(const char *crs);

duckdb_state duckdb_vx_register_scan_replacement(duckdb_database duckdb_database);

/// Creates a new reusable dictionary from a logical type and size.
duckdb_vx_reusable_dict duckdb_vx_reusable_dict_create(duckdb_logical_type logical_type, idx_t size);

/// Destroys the reusable dictionary.
void duckdb_vx_reusable_dict_destroy(duckdb_vx_reusable_dict *dict);

/// Clones the reusable dictionary.
duckdb_vx_reusable_dict duckdb_vx_reusable_dict_clone(duckdb_vx_reusable_dict dict);

/// Get the internal vector of the reusable dictionary.
void duckdb_vx_reusable_dict_set_vector(duckdb_vx_reusable_dict reusable, duckdb_vector *out_vector);

/// Creates a dictionary vector using a reusable dictionary and a selection vector.
void duckdb_vx_vector_dictionary_reusable(duckdb_vector vector,
                                          duckdb_vx_reusable_dict reusable,
                                          duckdb_selection_vector sel_vec);

// Create a null value with a reference to a logical type.
duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty);

/// Creates a GEOMETRY value containing the given WKB bytes and CRS.
duckdb_value duckdb_vx_value_create_geometry(const uint8_t *wkb, idx_t len, const char *crs);

/// Extracts the raw WKB bytes from a GEOMETRY value as a duckdb_blob.
/// The returned `data` pointer must be freed with `duckdb_free`. Returns `{nullptr, 0}`
/// if `value` is null or not a GEOMETRY value.
duckdb_blob duckdb_vx_value_get_geometry(duckdb_value value);

#ifdef __cplusplus
}
#endif
