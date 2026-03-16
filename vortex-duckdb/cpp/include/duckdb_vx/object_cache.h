// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/client_context.h"

#pragma once

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_object_cache_ *duckdb_vx_object_cache;

duckdb_vx_object_cache duckdb_client_context_get_object_cache(duckdb_client_context context);

// Function pointer type for custom deleter
typedef void (*duckdb_vx_deleter_fn)(void *ptr);

// Writes the `value` to the object cache with the key `key`, overwriting the current value if it exists.
void duckdb_vx_object_cache_put(duckdb_vx_object_cache object_cache,
                                const char *key,
                                void *value,
                                uint64_t estimated_size,
                                duckdb_vx_deleter_fn deleter);

// Fetches the key from the object cache, returning nullptr if the key is not present.
void *duckdb_vx_object_cache_get(duckdb_vx_object_cache object_cache, const char *key);

#ifdef __cplusplus /* End C ABI */
}
#endif
