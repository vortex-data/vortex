#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

char * duckdb_vx_logical_type_stringify(duckdb_logical_type ty);

#ifdef __cplusplus /* End C ABI */
}
#endif
