// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
DUCKDB_INCLUDES_END

#include "duckdb_vx/data.h"
#include "duckdb_vx/error.h"
#include "duckdb_vx/vector_buffer.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

/// Slice the vector to a new dictionary vector, using the current vector's values and
/// the provided selection vector.
///
/// A dictionary slice holds a strong reference to all memory it uses.
void duckdb_vx_vector_slice_to_dictionary(duckdb_vector ffi_vector,
                                          duckdb_selection_vector selection_vector,
                                          idx_t selection_vector_length);

/// Creates a dictionary vector for a given values vector and selection vector.
///
/// A dictionary holds a strong reference to all memory it uses.
///
/// `dictionary` differs from `slice_to_dictionary` in that it initializes hash caching:
/// https://github.com/duckdb/duckdb/blob/0dcf633f603a629981d089202f93b9080cb1a3e9/src/common/types/vector.cpp#L293
void duckdb_vx_vector_dictionary(duckdb_vector ffi_vector,
                                 duckdb_vector ffi_dict,
                                 idx_t dictionary_size,
                                 duckdb_selection_vector ffi_sel_vec,
                                 idx_t count);

void duckdb_vx_set_dictionary_vector_length(duckdb_vector dict, unsigned int len);

// Add the buffer to the string vector (basically, keep it alive as long as the vector).
void duckdb_vx_string_vector_add_vector_data_buffer(duckdb_vector ffi_vector, duckdb_vx_vector_buffer buffer);

// Add the buffer to the data vector (basically, keep it alive as long as the vector) and set the data
// pointer. You must ensure that the ptr is valid for the lifetime of the vector and the ptr addr + size is
// valid.
void duckdb_vx_vector_set_vector_data_buffer(duckdb_vector ffi_vector, duckdb_vx_vector_buffer buffer);

// Reset vector's validity mask to nullptr, making all vector's elements valid.
// vector must not be a DictionaryVector or a SequenceVector
void duckdb_vx_vector_set_all_valid(duckdb_vector ffi_vector);

// Set the data pointer for the vector. This is the start of the values array in the vector.
void duckdb_vx_vector_set_data_ptr(duckdb_vector ffi_vector, void *ptr);

// Set the validity pointer for the vector to external data, and store the buffer in auxiliary
// to keep it alive. The validity pointer is derived from data_ptr at the given u64 offset.
// The buffer is attached purely as a keep-alive. This enables zero-copy export of validity masks.
void duckdb_vx_vector_set_validity_data(duckdb_vector ffi_vector,
                                        idx_t u64_offset,
                                        idx_t capacity,
                                        duckdb_vx_vector_buffer buffer,
                                        void *data_ptr);

// Converts a duckdb flat vector into a Sequence vector.
void duckdb_vx_sequence_vector(duckdb_vector c_vector, int64_t start, int64_t step, idx_t capacity);

void duckdb_vector_flatten(duckdb_vector vector, unsigned long len);

const char *duckdb_vector_to_string(duckdb_vector vector, unsigned long len, duckdb_vx_error *err);

duckdb_value duckdb_vx_vector_get_value(duckdb_vector ffi_vector, idx_t index);

#ifdef __cplusplus /* End C ABI */
}
#endif
