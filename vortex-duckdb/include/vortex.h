// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//

// clang-format off

#include "duckdb.h"


#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern void vortex_init_rust(duckdb_database db);

extern const char *vortex_version_rust(void);

extern const char *vortex_extension_version_rust(void);

extern void duckdb_table_function_to_string(void *bind_data, duckdb_vx_string_map map);

extern
bool duckdb_table_function_statistics(const void *bind_data,
                                      size_t column_index,
                                      duckdb_column_statistics *stats_out);

extern double duckdb_table_function_scan_progress(void *global_state);

extern
void duckdb_table_function_get_partition_data(void *global_init_data,
                                              void *local_init_data,
                                              duckdb_vx_partition_data *partition_data_out);

extern
bool duckdb_table_function_pushdown_complex_filter(void *bind_data,
                                                   duckdb_vx_expr expr,
                                                   duckdb_vx_error *error_out);

extern
bool duckdb_table_function_pushdown_projection_expression(void *bind_data,
                                                          duckdb_vx_expr expr,
                                                          size_t column_id,
                                                          duckdb_vx_error *error_out);

extern
void duckdb_table_function_scan(void *global_init_data,
                                void *local_init_data,
                                duckdb_data_chunk output,
                                duckdb_vx_error *error_out);

extern bool duckdb_table_function_pushdown_expression(duckdb_vx_expr expr);

extern
void duckdb_table_function_cardinality(void *bind_data,
                                       duckdb_vx_node_statistics *node_stats_out);

extern
duckdb_vx_data duckdb_table_function_init_global(const duckdb_vx_tfunc_init_input *init_input,
                                                 duckdb_vx_error *error_out);

extern duckdb_vx_data duckdb_table_function_init_local(void *global_init_data);

extern
duckdb_vx_data duckdb_table_function_bind(duckdb_client_context ctx,
                                          duckdb_vx_tfunc_bind_input bind_input,
                                          duckdb_vx_tfunc_bind_result bind_result,
                                          duckdb_vx_error *error_out);

extern duckdb_vx_data duckdb_table_function_bind_data_clone(const void *bind_data);

extern
duckdb_vx_data duckdb_copy_function_copy_to_bind(const char *const *column_names,
                                                 size_t column_name_count,
                                                 const duckdb_logical_type *column_types,
                                                 size_t column_type_count,
                                                 duckdb_vx_error *error_out);

extern
duckdb_vx_data duckdb_copy_function_copy_to_initialize_global(duckdb_client_context client_context,
                                                              const void *bind_data,
                                                              const char *file_path,
                                                              duckdb_vx_error *error_out);

extern
void duckdb_copy_function_copy_to_sink(const void *bind_data,
                                       void *global_data,
                                       duckdb_data_chunk data_chunk,
                                       duckdb_vx_error *error_out);

extern void duckdb_copy_function_copy_to_finalize(void *global_data, duckdb_vx_error *error_out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

// clang-format on
