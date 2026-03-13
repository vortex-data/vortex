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

typedef struct duckdb_vx_reusable_dict_ *duckdb_vx_reusable_dict;

/// Creates a new reusable dictionary from a logical type and size.
/// The returned dictionary can be used with duckdb_vx_vector_dictionary_reusable.
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

#ifdef __cplusplus /* End C ABI */
}
#endif
