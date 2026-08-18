// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "duckdb.h"
#include "table_filter.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct duckdb_bind_result_ *duckdb_bind_result;

// Add a result column to the bind info.
void duckdb_vx_tfunc_bind_result_add_column(duckdb_bind_result ffi_result,
                                            const char *name_str,
                                            size_t name_len,
                                            duckdb_logical_type ffi_type);

typedef struct duckdb_vx_string_map_ *duckdb_vx_string_map;
// Add a key-value pair to the string map
void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value);

// Input data passed into the init_global and init_local callbacks.
typedef struct {
    const void *bind_data;
    idx_t *column_ids;
    size_t column_ids_count;
    duckdb_vx_table_filter_set filters;
    duckdb_client_context client_context;
} duckdb_vx_tfunc_init_input;

// Result data returned from the cardinality callback.
typedef struct {
    idx_t estimated_cardinality;
    bool has_estimated_cardinality;
    idx_t max_cardinality;
    bool has_max_cardinality;
} duckdb_vx_node_statistics;

typedef struct {
    // Set only for strings and primitive types
    duckdb_value min;
    duckdb_value max;
    // upper bit: "length is set". lower 32 bits: DuckDB's max string length.
    // set only for strings
    uint64_t max_string_length;
    bool has_null;
    // owned column type
    duckdb_logical_type type;
} duckdb_column_statistics;

duckdb_state duckdb_vx_register_table_functions(duckdb_database ffi_db);

typedef struct duckdb_vx_agg_input_ *duckdb_vx_agg_input;
idx_t duckdb_vx_aggregate_len(duckdb_vx_agg_input ffi);
duckdb_vx_expr duckdb_vx_aggregate_at(duckdb_vx_agg_input ffi, idx_t index, idx_t *proj_idx);

#ifdef __cplusplus
}
#endif
