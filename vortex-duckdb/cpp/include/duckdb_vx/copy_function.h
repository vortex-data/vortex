// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * We redefine a C API for DuckDB Copy Functions used to copy/write values from duckdb to a vortex file.
 * See table_filter.h for more info
 */
#pragma once

#include "data.h"
#include "error.h"
#include "logical_type.h"
#include "duckdb_vx/data.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Info passed into the bind callback. The callback should set error or else add result columns.
typedef struct duckdb_vx_copy_func_bind_input_ *duckdb_vx_copy_func_bind_input;

// Input data passed into the init_global and init_local callbacks.
typedef struct {
    const void *bind_data;
    const void *local_state;
    const void *global_state;
} duckdb_vx_copy_func_init_input;

// TODO(joe): expose via c api
// typedef enum copy_function_execution_mode_ {
// REGULAR_COPY_TO_FILE = 1,
// PARALLEL_COPY_TO_FILE,
// BATCH_COPY_TO_FILE
// } copy_function_execution_mode;

// A transparent DuckDB copy function vtable, which can be used to configure a copy function.
typedef struct {
    // The name of the copy function.
    const char *name;

    // The extension of the files written by the copy function.
    const char *extension;

    duckdb_vx_data (*bind)(duckdb_vx_copy_func_bind_input input,
                           const char *const *column_names,
                           unsigned long column_name_count,
                           const duckdb_logical_type *column_types,
                           unsigned long column_type_count,
                           duckdb_vx_error *error_out);

    duckdb_vx_data (*init_global)(duckdb_client_context ctx,
                                  const void *bind_data,
                                  const char *file_path,
                                  duckdb_vx_error *error_out);

    duckdb_vx_data (*init_local)(const void *bind_data, duckdb_vx_error *error_out);

    void (*copy_to_sink)(const void *bind_data,
                         void *global_data,
                         void *local_data,
                         duckdb_data_chunk data_chunk_out,
                         duckdb_vx_error *error_out);

    void (*copy_to_finalize)(const void *bind_data, void *global_data, duckdb_vx_error *error_out);

    // TODO(joe): expose via c api
    // copy_function_execution_mode (*execution_mode)(bool preserve_insertion_order, bool
    // supports_batch_index);
} duckdb_vx_copy_func_vtab_t;

// Due to a limitation in the copy function duckdb api we have to have global (copy) vtabs.
duckdb_vx_copy_func_vtab_t *get_vtab_one();

// A single function for configuring the DuckDB table function vtable.
duckdb_state duckdb_vx_copy_func_register_vtab_one(duckdb_database ffi_db);

#ifdef __cplusplus /* End C ABI */
};
#endif