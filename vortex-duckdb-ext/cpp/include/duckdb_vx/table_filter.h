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
	DUCKDB_VX_TABLE_FILTER_TYPE_EXPRESSION_FILTER = 9 // an arbitrary expression
} duckdb_vx_table_filter_type;

typedef struct duckdb_vx_table_filter_set_ *duckdb_vx_table_filter_set;

typedef struct duckdb_vx_table_filter_ *duckdb_vx_table_filter;

typedef struct duckdb_vx_dynamic_filter_data_ *duckdb_vx_dynamic_filter_data;

/// Returns nullable ptr if there is no filter for the column index.
duckdb_vx_table_filter duckdb_vx_table_filter_set_get(duckdb_vx_table_filter_set filter_set,
                                                      idx_t column_index);

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

duckdb_vx_dynamic_filter_data duckdb_vx_table_filter_get_dynamic(duckdb_vx_table_filter ffi_filter);

#ifdef __cplusplus /* End C ABI */
}
#endif
