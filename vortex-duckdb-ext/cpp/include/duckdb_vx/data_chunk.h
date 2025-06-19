#pragma once

#include "duckdb.h"
#include "error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// TODO(joe): remove 2 once we are off the duckdb fork
const char *duckdb_data_chunk_to_string2(duckdb_data_chunk chunk, duckdb_vx_error *err);

// TODO(joe): remove 2 once we are off the duckdb fork
void duckdb_data_chunk_verify2(duckdb_data_chunk chunk, duckdb_vx_error *err);

#ifdef __cplusplus /* End C ABI */
}
#endif
