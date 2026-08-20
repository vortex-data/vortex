// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//

// clang-format off

#include "duckdb.h"


#pragma once

#define COUNT_STAR_PROJ_IDX UINT64_MAX

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern void duckdb_table_function_to_string(const void *bind, duckdb_vx_string_map map);

extern
bool duckdb_table_function_pushdown_complex_filter(void *bind,
                                                   duckdb_vx_expr expr,
                                                   duckdb_vx_error *error);

extern
bool duckdb_table_function_pushdown_projection_expression(void *bind,
                                                          duckdb_vx_expr expr,
                                                          size_t column_id,
                                                          duckdb_vx_error *error);

extern
bool duckdb_table_function_pushdown_projection_aggregates(void *bind,
                                                          duckdb_vx_agg_input input,
                                                          duckdb_vx_error *error);

extern bool duckdb_table_function_pushdown_expression(duckdb_vx_expr expr);

extern
void duckdb_table_function_cardinality(const void *bind,
                                       uint64_t file_count,
                                       duckdb_vx_node_statistics *stats);

extern
duckdb_vx_data duckdb_table_function_init_global(const duckdb_vx_tfunc_init_input *init_input,
                                                 duckdb_vx_error *error);

extern duckdb_vx_data duckdb_table_function_init_local(const void *bind, const void *global);

extern
duckdb_vx_data duckdb_reader_bind(const void *first_file,
                                  duckdb_bind_result result,
                                  duckdb_vx_error *error_out);

extern
duckdb_vx_data duckdb_reader_open(const char *file_path,
                                  size_t file_path_len,
                                  duckdb_vx_error *error);

extern
bool duckdb_reader_get_statistics(const void *file,
                                  const void *bind,
                                  const char *column_name,
                                  size_t column_name_len,
                                  duckdb_column_statistics *stats_out);

extern bool duckdb_reader_initialize(const void *global, void *file, duckdb_vx_error *error);

extern duckdb_logical_type duckdb_reader_bind_column_type(const void *bind, size_t index);

extern bool duckdb_reader_is_aggregate(const void *bind);

extern bool duckdb_reader_try_initialize_scan(void *local, void *file);

extern
bool duckdb_reader_scan(const void *file,
                        const void *global,
                        void *local,
                        duckdb_data_chunk chunk,
                        duckdb_vx_error *error);

extern double duckdb_reader_get_progress_in_file(const void *file);

extern
bool duckdb_reader_finalize_scan(const void *global,
                                 duckdb_data_chunk chunk,
                                 duckdb_vx_error *error);

extern void duckdb_reader_finish_reading(const void *global, void *local);

extern duckdb_vx_data duckdb_table_function_bind_data_clone(const void *bind);

extern
duckdb_vx_data duckdb_copy_function_copy_to_bind(const char *const *column_names,
                                                 size_t column_name_count,
                                                 const duckdb_logical_type *column_types,
                                                 size_t column_type_count,
                                                 duckdb_vx_error *error_out);

extern
duckdb_vx_data duckdb_copy_function_copy_to_initialize_global(const void *bind_data,
                                                              const char *file_path,
                                                              duckdb_vx_error *error_out);

extern
void duckdb_copy_function_copy_to_sink(const void *bind_data,
                                       const void *global_data,
                                       duckdb_data_chunk data_chunk,
                                       duckdb_vx_error *error_out);

extern void duckdb_copy_function_copy_to_finalize(void *global_data, duckdb_vx_error *error_out);

extern duckdb_vx_data duckdb_copy_function_prepare_batch_new(void);

extern
void duckdb_copy_function_prepare_batch_push(const void *bind,
                                             void *batch,
                                             duckdb_data_chunk chunk,
                                             duckdb_vx_error *error);

extern
void duckdb_copy_function_flush_batch(const void *global,
                                      const void *batch,
                                      duckdb_vx_error *error);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

// clang-format on
