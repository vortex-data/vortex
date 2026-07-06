// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "duckdb.h"
#include "table_filter.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Info passed into the bind callback. The callback should set error or else add result columns.
typedef struct duckdb_vx_tfunc_bind_input_ *duckdb_vx_tfunc_bind_input;
typedef struct duckdb_vx_tfunc_bind_result_ *duckdb_vx_tfunc_bind_result;

// Fetch a parameter from the bind info.
// The caller is responsible for freeing the value using duckdb_value_free.
duckdb_value duckdb_vx_tfunc_bind_input_get_parameter(duckdb_vx_tfunc_bind_input ffi_input, size_t index);

// Add a result column to the bind info.
void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result,
                                            const char *name_str,
                                            size_t name_len,
                                            duckdb_logical_type ffi_type);

typedef struct duckdb_vx_string_map_ *duckdb_vx_string_map;
// Add a key-value pair to the string map
void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value);

// Input data passed into the init_global and init_local callbacks.
typedef struct {
    const void *bind_data;

    /**
     * Projected columns that are requested to be read. These are not
     * all columns, only the ones DuckDB optimizer thinks we should read.
     */
    idx_t *column_ids;
    size_t column_ids_count;

    /**
     * Post filter projected columns. Our table function implements filter
     * pushdown so this list is a subset of columns referenced in column_ids
     * after filter pushdown and filter pruning. May be empty, in which case
     * column_ids should be used.
     * Indices in this list reference values from column_ids. I.e. if
     * column_ids=[1,5,6], projection_ids=[1], output column should be
     * column_ids[1] = 5
     *
     * Example usage:
     * https://github.com/duckdb/duckdb/blob/dc11eadd8f0a7c600f0034810706605ebe10d5b9/src/include/duckdb/function/table_function.hpp#L147
     */
    const idx_t *projection_ids;
    size_t projection_ids_count;

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
} duckdb_column_statistics;

const idx_t INVALID_IDX = UINT64_MAX;

typedef struct {
    idx_t partition_index;
    // Either INVALID_IDX or position of column in output for file_index column
    size_t file_index_column_pos;
    // File index for the exported partition.
    size_t file_index;
} duckdb_vx_partition_data;

duckdb_state duckdb_vx_register_table_functions(duckdb_database ffi_db);

#ifdef __cplusplus
}
#endif
