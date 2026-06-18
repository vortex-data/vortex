// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#include "duckdb_vx/expr.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef enum DUCKDB_VX_TABLE_FILTER_TYPE {
    DUCKDB_VX_TABLE_FILTER_TYPE_CONSTANT_COMPARISON = 0, // constant comparison (e.g. =C, >C, >=C, <C, <=C)
    DUCKDB_VX_TABLE_FILTER_TYPE_IS_NULL = 1,             // C IS NULL
    DUCKDB_VX_TABLE_FILTER_TYPE_IS_NOT_NULL = 2,         // C IS NOT NULL
    DUCKDB_VX_TABLE_FILTER_TYPE_CONJUNCTION_OR = 3,      // OR of different filters
    DUCKDB_VX_TABLE_FILTER_TYPE_CONJUNCTION_AND = 4,     // AND of different filters
    DUCKDB_VX_TABLE_FILTER_TYPE_STRUCT_EXTRACT = 5,      // filter applies to child-column of struct
    DUCKDB_VX_TABLE_FILTER_TYPE_OPTIONAL_FILTER = 6, // executing filter is not required for query correctness
    DUCKDB_VX_TABLE_FILTER_TYPE_IN_FILTER = 7,       // col IN (C1, C2, C3, ...)
    DUCKDB_VX_TABLE_FILTER_TYPE_DYNAMIC_FILTER = 8,  // dynamic filters can be updated at run-time
    DUCKDB_VX_TABLE_FILTER_TYPE_EXPRESSION_FILTER = 9, // an arbitrary expression
    DUCKDB_VX_TABLE_FILTER_TYPE_BLOOM_FILTER =
        10 // a probabilistic filter that can test whether a value is in a set of other value
} duckdb_vx_table_filter_type;

typedef struct duckdb_vx_table_filter_set_ *duckdb_vx_table_filter_set;

typedef struct duckdb_vx_table_filter_ *duckdb_vx_table_filter;

/// If a table filter with position idx exists, return its column index,
/// assign the filter to table_filter_out.
/// If such filter doesn't exist, set table_filter_out to nullptr and return 0
/// idx is a position in a filter array sorted by increasing column indices.
///
/// TODO(myrrc) is idx really filter's position or it's a column's idx?
idx_t duckdb_vx_table_filter_set_get(duckdb_vx_table_filter_set filter_set,
                                     size_t idx,
                                     duckdb_vx_table_filter *table_filter_out);

duckdb_vx_table_filter_type duckdb_vx_table_filter_get_type(duckdb_vx_table_filter ffi_filter);

const char *duckdb_vx_table_filter_to_debug_string(duckdb_vx_table_filter ffi_filter);

idx_t duckdb_vx_table_filter_set_size(duckdb_vx_table_filter_set ffi_filter_set);

typedef struct {
    duckdb_value value;
    duckdb_vx_expr_type comparison_type;
} duckdb_vx_table_filter_constant;

void duckdb_vx_table_filter_get_constant(duckdb_vx_table_filter ffi_filter,
                                         duckdb_vx_table_filter_constant *out);

typedef struct {
    duckdb_vx_table_filter *children;
    size_t children_count;
} duckdb_vx_table_filter_conjunction;

void duckdb_vx_table_filter_get_conjunction_or(duckdb_vx_table_filter ffi_filter,
                                               duckdb_vx_table_filter_conjunction *out);

void duckdb_vx_table_filter_get_conjunction_and(duckdb_vx_table_filter ffi_filter,
                                                duckdb_vx_table_filter_conjunction *out);

typedef struct duckdb_vx_dynamic_filter_data_ *duckdb_vx_dynamic_filter_data;
void duckdb_vx_dynamic_filter_data_free(duckdb_vx_dynamic_filter_data *ffi_data);
duckdb_value duckdb_vx_dynamic_filter_data_get_value(duckdb_vx_dynamic_filter_data ffi_data);

typedef struct {
    duckdb_vx_dynamic_filter_data data;
    duckdb_vx_expr_type comparison_type;
} duckdb_vx_table_filter_dynamic;

void duckdb_vx_table_filter_get_dynamic(duckdb_vx_table_filter ffi_filter,
                                        duckdb_vx_table_filter_dynamic *out);

duckdb_vx_table_filter duckdb_vx_table_filter_get_optional(duckdb_vx_table_filter ffi_filter);

duckdb_vx_expr duckdb_vx_table_filter_get_expression(duckdb_vx_table_filter ffi_filter);

typedef struct {
    duckdb_vx_table_filter child_filter;
    char *child_name;
    size_t child_name_len;
} duckdb_vx_table_filter_struct_extract;

void duckdb_vx_table_filter_get_struct_extract(duckdb_vx_table_filter ffi_filter,
                                               duckdb_vx_table_filter_struct_extract *out);

// Wrapper around a vector<Value>. The C API only knows about duckdb_value, which is itself a ptr to a Value,
// so we cannot simply unwrap a vector<Value> on the Rust side (we would need a vector<*Value>).
typedef struct duckdb_vx_values_vec_ *duckdb_vx_values_vec;

duckdb_value duckdb_vx_values_vec_get(duckdb_vx_values_vec ffi_vec, size_t idx);

typedef struct {
    duckdb_vx_values_vec values;
    size_t values_count;
} duckdb_vx_table_filter_in_filter;

void duckdb_vx_table_filter_get_in_filter(duckdb_vx_table_filter ffi_filter,
                                          duckdb_vx_table_filter_in_filter *out);

#ifdef __cplusplus /* End C ABI */
}
#endif
