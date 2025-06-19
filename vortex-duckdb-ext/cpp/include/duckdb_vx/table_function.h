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

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Info passed into the bind callback. The callback should set error or else add result columns.
typedef struct duckdb_vx_tfunc_bind_input_ *duckdb_vx_tfunc_bind_input;
typedef struct duckdb_vx_tfunc_bind_result_ *duckdb_vx_tfunc_bind_result;

// Fetch the parameter count from the bind info.
size_t duckdb_vx_tfunc_bind_input_get_parameter_count(duckdb_vx_tfunc_bind_input ffi_input);

// Fetch a parameter from the bind info.
// The caller is responsible for freeing the value using duckdb_value_free.
duckdb_value duckdb_vx_tfunc_bind_input_get_parameter(duckdb_vx_tfunc_bind_input ffi_input, size_t index);

duckdb_value duckdb_vx_tfunc_bind_input_get_named_parameter(duckdb_vx_tfunc_bind_input ffi_input,
                                                            const char *name_str);

// Add a result column to the bind info.
void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result, const char *name_str,
                                            size_t name_len, duckdb_logical_type ffi_type);

// Input data passed into the init_global and init_local callbacks.
typedef struct {
    const void *bind_data;
    idx_t *column_ids;
    size_t column_ids_count;
    // uint64_t *column_indexes;
    // size_t column_indexes_count;
    const idx_t *projection_ids;
    size_t projection_ids_count;
    duckdb_vx_table_filter_set filters;
    // void *sample_options;
} duckdb_vx_tfunc_init_input;

// A transparent DuckDB table function vtable, which can be used to configure a table function.
// See duckdb/include/function/tfunc.hpp for details on each field.
typedef struct {
    // The name of the table function.
    const char *name;

    // The parameters of the table function.
    const duckdb_logical_type *parameters;
    size_t parameter_count;

    // The named parameters of the table function.
    const duckdb_logical_type *named_parameter_types;
    const char *const *named_parameter_names;
    size_t named_parameter_count;

    duckdb_vx_data (*bind)(duckdb_vx_tfunc_bind_input input, duckdb_vx_tfunc_bind_result result,
                           duckdb_vx_error *error_out);
    duckdb_vx_data (*bind_data_clone)(const void *bind_data, duckdb_vx_error *error_out);

    // void *bind_replace;
    // void *bind_operator;

    duckdb_vx_data (*init_global)(const duckdb_vx_tfunc_init_input *input, duckdb_vx_error *error_out);

    duckdb_vx_data (*init_local)(const duckdb_vx_tfunc_init_input *input, void *init_global_data,
                                 duckdb_vx_error *error_out);

    void (*function)(const void *bind_data, void *init_global_data, void *init_local_data,
                     duckdb_data_chunk data_chunk_out, duckdb_vx_error *error_out);

    // void *in_out_function;
    // void *in_out_function_final;
    void *statistics;
    // void *dependency;
    void *cardinality;

    bool (*pushdown_complex_filter)(void *bind_data, duckdb_vx_expr expr, duckdb_vx_error *error_out);

    void *pushdown_expression;
    // void *to_string;
    // void *dynamic_to_string;
    void *table_scan_progress;
    // void *get_partition_data;
    // void *get_bind_info;
    // void *type_pushdown;
    // void *get_multi_file_reader;
    // void *supports_pushdown_type;
    // void *get_partition_info;
    // void *get_partition_stats;
    // void *get_virtual_columns;
    // void *get_row_id_columns;

    bool projection_pushdown;
    bool filter_pushdown;
    bool filter_prune;
    bool sampling_pushdown;
    bool late_materialization;
} duckdb_vx_tfunc_vtab_t;

// A single function for configuring the DuckDB table function vtable.
duckdb_state duckdb_vx_tfunc_register(duckdb_connection ffi_conn, const duckdb_vx_tfunc_vtab_t *vtab);

#ifdef __cplusplus /* End C ABI */
}
#endif
