// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include <duckdb.h>
DUCKDB_INCLUDES_END

#ifdef __cplusplus
extern "C" {
#endif

typedef struct duckdb_vx_object_cache_entry_ *duckdb_vx_object_cache_entry;

duckdb_vx_object_cache_entry duckdb_vx_object_cache_get(duckdb_client_context ctx,
                                                         const char *key,
                                                         size_t key_len,
                                                         const char *object_type);

void *duckdb_vx_object_cache_entry_get_data(duckdb_vx_object_cache_entry entry);

void duckdb_vx_object_cache_entry_free(duckdb_vx_object_cache_entry *entry);

duckdb_state duckdb_vx_object_cache_put(duckdb_client_context ctx,
                                        const char *key,
                                        size_t key_len,
                                        const char *object_type,
                                        idx_t estimated_memory,
                                        void *data,
                                        duckdb_delete_callback_t delete_callback);

#ifdef __cplusplus
}
#endif
