#pragma once

#include "duckdb.h"
#include "duckdb_vx/data.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

/// Slice to a dictionary vector, takes ownership of selection vector.
void duckdb_vx_vector_slice_to_dictionary(duckdb_vector ffi_vector, duckdb_selection_vector selection_vector,
                                          idx_t selection_vector_length);

// Add the buffer to the string vector (basically, keep it alive as long as the vector).
void duckdb_vx_string_vector_add_buffer(duckdb_vector ffi_vector, duckdb_vx_data buffer);

#ifdef __cplusplus /* End C ABI */
}
#endif
