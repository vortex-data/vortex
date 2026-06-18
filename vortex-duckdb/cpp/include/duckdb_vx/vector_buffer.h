// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#include "duckdb_vx/data.h"
#include "duckdb_vx/error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_vector_buffer_ *duckdb_vx_vector_buffer;

// Create a external vector buffer from an existing data buffer, used to allow DuckDB to keep a reference to
// the buffer.
duckdb_vx_vector_buffer duckdb_vx_vector_buffer_create(duckdb_vx_data buffer);

// Destroy the vector buffer.
void duckdb_vx_vector_buffer_destroy(duckdb_vx_vector_buffer *buffer);

#ifdef __cplusplus /* End C ABI */
}
#endif
