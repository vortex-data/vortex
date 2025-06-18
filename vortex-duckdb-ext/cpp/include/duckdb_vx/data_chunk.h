#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

const char *duckdb_data_chunk_to_string(duckdb_data_chunk chunk);

void duckdb_data_chunk_verify(duckdb_data_chunk chunk);

#ifdef __cplusplus /* End C ABI */
}
#endif
