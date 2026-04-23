// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * We redefine a C API for DuckDB Table Functions in order to expose the full functionality of the C++ API.
 *
 * Since this C API has no stability requirements (it's versioned lock-step with the Rust bindings), we can
 * take a transparent vtable struct to populate the C++ Table Function vtable.
 */
#pragma once

#include "error.h"
#include "table_filter.h"
#include "duckdb_vx/data.h"

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
    idx_t max_cardinality;
    bool has_estimated_cardinality;
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

// vtable mimicking subset of TableFunction.
// See duckdb/include/function/tfunc.hpp
typedef struct {
    const char *name;
    const duckdb_logical_type *parameters;
    size_t parameter_count;

    duckdb_vx_data (*bind)(duckdb_client_context ctx,
                           duckdb_vx_tfunc_bind_input input,
                           duckdb_vx_tfunc_bind_result result,
                           duckdb_vx_error *error_out);

    duckdb_vx_data (*bind_data_clone)(const void *bind_data, duckdb_vx_error *error_out);

    duckdb_vx_data (*init_global)(const duckdb_vx_tfunc_init_input *input, duckdb_vx_error *error_out);

    duckdb_vx_data (*init_local)(const duckdb_vx_tfunc_init_input *input,
                                 void *init_global_data,
                                 duckdb_vx_error *error_out);

    void (*function)(duckdb_client_context ctx,
                     const void *bind_data,
                     void *init_global_data,
                     void *init_local_data,
                     duckdb_data_chunk data_chunk_out,
                     duckdb_vx_error *error_out);

    bool (*statistics)(duckdb_client_context context,
                       const void *bind_data,
                       size_t column_index,
                       duckdb_column_statistics *stats_out);

    void (*cardinality)(void *bind_data, duckdb_vx_node_statistics *node_stats_out);

    bool (*pushdown_complex_filter)(void *bind_data, duckdb_vx_expr expr, duckdb_vx_error *error_out);

    void (*to_string)(void *bind_data, duckdb_vx_string_map map);

    double (*table_scan_progress)(duckdb_client_context ctx, void *bind_data, void *global_state);

    idx_t (*get_partition_data)(const void *bind_data,
                                void *init_global_data,
                                void *init_local_data,
                                duckdb_vx_error *error_out);
} duckdb_vx_tfunc_vtab_t;

// A single function for configuring the DuckDB table function vtable.
duckdb_state duckdb_vx_tfunc_register(duckdb_database ffi_db, const duckdb_vx_tfunc_vtab_t *vtab);

#ifdef __cplusplus
}
#endif
