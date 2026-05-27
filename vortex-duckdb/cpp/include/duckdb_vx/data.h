// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Opaque data object with a deletion callback.
typedef struct duckdb_vx_data_ *duckdb_vx_data;

// Create an opaque data object with a delete callback.
duckdb_vx_data duckdb_vx_data_create(void *data_ptr, duckdb_delete_callback_t delete_callback);

#ifdef __cplusplus /* End C ABI */
}
#endif
