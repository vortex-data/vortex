#pragma once

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

/// Wrapper around FunctionInfo from .
typedef struct duckdb_vx_func_info_ *duckdb_vx_func_info;

#ifdef __cplusplus /* End C ABI */
}
#endif
