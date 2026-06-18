// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#include "error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

const char *duckdb_data_chunk_to_string(duckdb_data_chunk chunk, duckdb_vx_error *err);

void duckdb_data_chunk_verify(duckdb_data_chunk chunk, duckdb_vx_error *err);

#ifdef __cplusplus /* End C ABI */
}
#endif
