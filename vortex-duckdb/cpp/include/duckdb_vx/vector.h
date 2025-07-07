// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "duckdb_vx/data.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

/// Slice to a dictionary vector.
void duckdb_vx_vector_slice_to_dictionary(duckdb_vector ffi_vector, duckdb_selection_vector selection_vector,
                                          idx_t selection_vector_length);

void duckdb_vx_set_dictionary_vector_id(duckdb_vector dict, const char *id, unsigned int id_len);

void duckdb_vx_set_dictionary_vector_length(duckdb_vector dict, unsigned int len);

// Add the buffer to the string vector (basically, keep it alive as long as the vector).
void duckdb_vx_string_vector_add_buffer(duckdb_vector ffi_vector, duckdb_vx_data buffer);

// Converts a duckdb flat vector into a Sequence vector.
void duckdb_vx_sequence_vector(duckdb_vector c_vector, int64_t start, int64_t step, idx_t capacity);

void duckdb_vector_flatten(duckdb_vector vector, unsigned long len);

const char *duckdb_vector_to_string(duckdb_vector vector, unsigned long len, duckdb_vx_error *err);

#ifdef __cplusplus /* End C ABI */
}
#endif
