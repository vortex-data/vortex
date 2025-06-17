#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

char *duckdb_vx_logical_type_stringify(duckdb_logical_type ty);
duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty);
bool duckdb_vx_logical_type_eq(duckdb_logical_type ty1, duckdb_logical_type ty2);

#ifdef __cplusplus /* End C ABI */
}
#endif
