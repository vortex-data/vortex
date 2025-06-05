#pragma once

#include <stddef.h>

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_error_ *duckdb_vx_error;

duckdb_vx_error duckdb_vx_error_create(const char *message, size_t message_length);

#ifdef __cplusplus  /* End C ABI */
}
#endif
