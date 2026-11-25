// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TEST-ONLY FFI constructors for DuckDB TableFilter objects.
// Compiled and bound only in non-release profiles (see build.rs) so unit tests can
// directly exercise every TableFilter conversion path without shipping test symbols
// in release artifacts.

#pragma once

#include "table_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct duckdb_vx_test_dynamic_filter_data_ *duckdb_vx_test_dynamic_filter_data;

void duckdb_vx_test_table_filter_destroy(duckdb_vx_table_filter *filter);
void duckdb_vx_test_dynamic_filter_data_destroy(duckdb_vx_test_dynamic_filter_data *data);
void duckdb_vx_test_dynamic_filter_data_set_value(duckdb_vx_test_dynamic_filter_data data,
                                                  duckdb_value value);

duckdb_vx_table_filter duckdb_vx_test_table_filter_create_constant(duckdb_vx_expr_type comparison_type,
                                                                   duckdb_value value);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_is_null(void);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_is_not_null(void);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_conjunction_and(duckdb_vx_table_filter *children,
                                                                          size_t count);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_conjunction_or(duckdb_vx_table_filter *children,
                                                                         size_t count);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_struct_extract(idx_t child_idx,
                                                                         const char *child_name,
                                                                         duckdb_vx_table_filter child_filter);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_optional(duckdb_vx_table_filter child_filter);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_in(duckdb_value *values, size_t count);
duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_dynamic(duckdb_vx_expr_type comparison_type,
                                           duckdb_value initial_value,
                                           duckdb_vx_test_dynamic_filter_data *data_out);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_bloom(void);

duckdb_vx_table_filter duckdb_vx_test_table_filter_create_expr_comparison(duckdb_vx_expr_type comparison_type,
                                                                          duckdb_value value,
                                                                          bool wrap_optional);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_expr_in(duckdb_logical_type col_type,
                                                                  duckdb_value *values,
                                                                  size_t count,
                                                                  bool wrap_optional);
duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_expr_dynamic(duckdb_vx_expr_type comparison_type,
                                                duckdb_value initial_value,
                                                bool wrap_optional,
                                                duckdb_vx_test_dynamic_filter_data *data_out);
duckdb_vx_table_filter duckdb_vx_test_table_filter_create_expr_unsupported(void);

#ifdef __cplusplus
}
#endif
