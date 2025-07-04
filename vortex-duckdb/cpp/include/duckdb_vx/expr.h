// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "scalar_function.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_expr_ *duckdb_vx_expr;

/// Return the string representation of the expression. Must be freed with `duckdb_vx_free`.
const char *duckdb_vx_expr_to_string(duckdb_vx_expr expr);

void duckdb_vx_destroy_expr(duckdb_vx_expr *expr);

// See ExpressionClass in duckdb/include/duckdb/common/enums/expression_class.hpp
typedef enum DUCKDB_VX_EXPR_CLASS {
    DUCKDB_VX_EXPR_CLASS_INVALID = 0,
    //===--------------------------------------------------------------------===//
    // Parsed Expressions
    //===--------------------------------------------------------------------===//
    DUCKDB_VX_EXPR_CLASS_AGGREGATE = 1,
    DUCKDB_VX_EXPR_CLASS_CASE = 2,
    DUCKDB_VX_EXPR_CLASS_CAST = 3,
    DUCKDB_VX_EXPR_CLASS_COLUMN_REF = 4,
    DUCKDB_VX_EXPR_CLASS_COMPARISON = 5,
    DUCKDB_VX_EXPR_CLASS_CONJUNCTION = 6,
    DUCKDB_VX_EXPR_CLASS_CONSTANT = 7,
    DUCKDB_VX_EXPR_CLASS_DEFAULT = 8,
    DUCKDB_VX_EXPR_CLASS_FUNCTION = 9,
    DUCKDB_VX_EXPR_CLASS_OPERATOR = 10,
    DUCKDB_VX_EXPR_CLASS_STAR = 11,
    DUCKDB_VX_EXPR_CLASS_SUBQUERY = 13,
    DUCKDB_VX_EXPR_CLASS_WINDOW = 14,
    DUCKDB_VX_EXPR_CLASS_PARAMETER = 15,
    DUCKDB_VX_EXPR_CLASS_COLLATE = 16,
    DUCKDB_VX_EXPR_CLASS_LAMBDA = 17,
    DUCKDB_VX_EXPR_CLASS_POSITIONAL_REFERENCE = 18,
    DUCKDB_VX_EXPR_CLASS_BETWEEN = 19,
    DUCKDB_VX_EXPR_CLASS_LAMBDA_REF = 20,
    //===--------------------------------------------------------------------===//
    // Bound Expressions
    //===--------------------------------------------------------------------===//
    DUCKDB_VX_EXPR_CLASS_BOUND_AGGREGATE = 25,
    DUCKDB_VX_EXPR_CLASS_BOUND_CASE = 26,
    DUCKDB_VX_EXPR_CLASS_BOUND_CAST = 27,
    DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF = 28,
    DUCKDB_VX_EXPR_CLASS_BOUND_COMPARISON = 29,
    DUCKDB_VX_EXPR_CLASS_BOUND_CONJUNCTION = 30,
    DUCKDB_VX_EXPR_CLASS_BOUND_CONSTANT = 31,
    DUCKDB_VX_EXPR_CLASS_BOUND_DEFAULT = 32,
    DUCKDB_VX_EXPR_CLASS_BOUND_FUNCTION = 33,
    DUCKDB_VX_EXPR_CLASS_BOUND_OPERATOR = 34,
    DUCKDB_VX_EXPR_CLASS_BOUND_PARAMETER = 35,
    DUCKDB_VX_EXPR_CLASS_BOUND_REF = 36,
    DUCKDB_VX_EXPR_CLASS_BOUND_SUBQUERY = 37,
    DUCKDB_VX_EXPR_CLASS_BOUND_WINDOW = 38,
    DUCKDB_VX_EXPR_CLASS_BOUND_BETWEEN = 39,
    DUCKDB_VX_EXPR_CLASS_BOUND_UNNEST = 40,
    DUCKDB_VX_EXPR_CLASS_BOUND_LAMBDA = 41,
    DUCKDB_VX_EXPR_CLASS_BOUND_LAMBDA_REF = 42,
    //===--------------------------------------------------------------------===//
    // Miscellaneous
    //===--------------------------------------------------------------------===//
    DUCKDB_VX_EXPR_CLASS_BOUND_EXPRESSION = 50,
    DUCKDB_VX_EXPR_CLASS_BOUND_EXPANDED = 51
} duckdb_vx_expr_class;

// See ExpressionType in duckdb/include/duckdb/common/enums/expression_type.hpp
typedef enum DUCKDB_VX_EXPR_TYPE {
    DUCKDB_VX_EXPR_TYPE_INVALID = 0,

    // explicitly cast left as right (right is integer in ValueType enum)
    DUCKDB_VX_EXPR_TYPE_OPERATOR_CAST = 12,
    // logical not operator
    DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT = 13,
    // is null operator
    DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL = 14,
    // is not null operator
    DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL = 15,
    // unpack operator
    DUCKDB_VX_EXPR_TYPE_OPERATOR_UNPACK = 16,

    // -----------------------------
    // Comparison Operators
    // -----------------------------
    // equal operator between left and right
    DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL = 25,
    // compare initial boundary
    DUCKDB_VX_EXPR_TYPE_COMPARE_BOUNDARY_START = DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL,
    // inequal operator between left and right
    CDUCKDB_VX_EXPR_TYPE_OMPARE_NOTEQUAL = 26,
    // less than operator between left and right
    DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN = 27,
    // greater than operator between left and right
    DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN = 28,
    // less than equal operator between left and right
    DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO = 29,
    // greater than equal operator between left and right
    DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO = 30,
    // IN operator [left IN (right1, right2, ...)]
    DUCKDB_VX_EXPR_TYPE_COMPARE_IN = 35,
    // NOT IN operator [left NOT IN (right1, right2, ...)]
    DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_IN = 36,
    // IS DISTINCT FROM operator
    DUCKDB_VX_EXPR_TYPE_COMPARE_DISTINCT_FROM = 37,

    DUCKDB_VX_EXPR_TYPE_COMPARE_BETWEEN = 38,
    DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_BETWEEN = 39,
    // IS NOT DISTINCT FROM operator
    DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_DISTINCT_FROM = 40,
    // compare final boundary
    DUCKDB_VX_EXPR_TYPE_COMPARE_BOUNDARY_END = DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_DISTINCT_FROM,

    // -----------------------------
    // Conjunction Operators
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_CONJUNCTION_AND = 50,
    DUCKDB_VX_EXPR_TYPE_CONJUNCTION_OR = 51,

    // -----------------------------
    // Values
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_VALUE_CONSTANT = 75,
    DUCKDB_VX_EXPR_TYPE_VALUE_PARAMETER = 76,
    DUCKDB_VX_EXPR_TYPE_VALUE_TUPLE = 77,
    DUCKDB_VX_EXPR_TYPE_VALUE_TUPLE_ADDRESS = 78,
    DUCKDB_VX_EXPR_TYPE_VALUE_NULL = 79,
    DUCKDB_VX_EXPR_TYPE_VALUE_VECTOR = 80,
    DUCKDB_VX_EXPR_TYPE_VALUE_SCALAR = 81,
    DUCKDB_VX_EXPR_TYPE_VALUE_DEFAULT = 82,

    // -----------------------------
    // Aggregates
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_AGGREGATE = 100,
    DUCKDB_VX_EXPR_TYPE_BOUND_AGGREGATE = 101,
    DUCKDB_VX_EXPR_TYPE_GROUPING_FUNCTION = 102,

    // -----------------------------
    // Window Functions
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_WINDOW_AGGREGATE = 110,

    DUCKDB_VX_EXPR_TYPE_WINDOW_RANK = 120,
    DUCKDB_VX_EXPR_TYPE_WINDOW_RANK_DENSE = 121,
    DUCKDB_VX_EXPR_TYPE_WINDOW_NTILE = 122,
    DUCKDB_VX_EXPR_TYPE_WINDOW_PERCENT_RANK = 123,
    DUCKDB_VX_EXPR_TYPE_WINDOW_CUME_DIST = 124,
    DUCKDB_VX_EXPR_TYPE_WINDOW_ROW_NUMBER = 125,

    DUCKDB_VX_EXPR_TYPE_WINDOW_FIRST_VALUE = 130,
    DUCKDB_VX_EXPR_TYPE_WINDOW_LAST_VALUE = 131,
    DUCKDB_VX_EXPR_TYPE_WINDOW_LEAD = 132,
    DUCKDB_VX_EXPR_TYPE_WINDOW_LAG = 133,
    DUCKDB_VX_EXPR_TYPE_WINDOW_NTH_VALUE = 134,

    // -----------------------------
    // Functions
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_FUNCTION = 140,
    DUCKDB_VX_EXPR_TYPE_BOUND_FUNCTION = 141,

    // -----------------------------
    // Operators
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_CASE_EXPR = 150,
    DUCKDB_VX_EXPR_TYPE_OPERATOR_NULLIF = 151,
    DUCKDB_VX_EXPR_TYPE_OPERATOR_COALESCE = 152,
    DUCKDB_VX_EXPR_TYPE_ARRAY_EXTRACT = 153,
    DUCKDB_VX_EXPR_TYPE_ARRAY_SLICE = 154,
    DUCKDB_VX_EXPR_TYPE_STRUCT_EXTRACT = 155,
    DUCKDB_VX_EXPR_TYPE_ARRAY_CONSTRUCTOR = 156,
    DUCKDB_VX_EXPR_TYPE_ARROW = 157,
    DUCKDB_VX_EXPR_TYPE_OPERATOR_TRY = 158,

    // -----------------------------
    // Subquery IN/EXISTS
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_SUBQUERY = 175,

    // -----------------------------
    // Parser
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_STAR = 200,
    DUCKDB_VX_EXPR_TYPE_TABLE_STAR = 201,
    DUCKDB_VX_EXPR_TYPE_PLACEHOLDER = 202,
    DUCKDB_VX_EXPR_TYPE_COLUMN_REF = 203,
    DUCKDB_VX_EXPR_TYPE_FUNCTION_REF = 204,
    DUCKDB_VX_EXPR_TYPE_TABLE_REF = 205,
    DUCKDB_VX_EXPR_TYPE_LAMBDA_REF = 206,

    // -----------------------------
    // Miscellaneous
    // -----------------------------
    DUCKDB_VX_EXPR_TYPE_CAST = 225,
    DUCKDB_VX_EXPR_TYPE_BOUND_REF = 227,
    DUCKDB_VX_EXPR_TYPE_BOUND_COLUMN_REF = 228,
    DUCKDB_VX_EXPR_TYPE_BOUND_UNNEST = 229,
    DUCKDB_VX_EXPR_TYPE_COLLATE = 230,
    DUCKDB_VX_EXPR_TYPE_LAMBDA = 231,
    DUCKDB_VX_EXPR_TYPE_POSITIONAL_REFERENCE = 232,
    DUCKDB_VX_EXPR_TYPE_BOUND_LAMBDA_REF = 233,
    DUCKDB_VX_EXPR_TYPE_BOUND_EXPANDED = 234
} duckdb_vx_expr_type;

duckdb_vx_expr_class duckdb_vx_expr_get_class(duckdb_vx_expr expr);

const char *duckdb_vx_expr_get_bound_column_ref_get_name(duckdb_vx_expr expr);

duckdb_value duckdb_vx_expr_bound_constant_get_value(duckdb_vx_expr expr);

typedef struct {
    duckdb_vx_expr left;
    duckdb_vx_expr right;
    duckdb_vx_expr_type type;
} duckdb_vx_expr_bound_comparison;

void duckdb_vx_expr_get_bound_comparison(duckdb_vx_expr expr, duckdb_vx_expr_bound_comparison *out);

typedef struct {
    duckdb_vx_expr *children;
    size_t children_count;
    duckdb_vx_expr_type type;
} duckdb_vx_expr_bound_conjunction;

void duckdb_vx_expr_get_bound_conjunction(duckdb_vx_expr expr, duckdb_vx_expr_bound_conjunction *out);

typedef struct {
    duckdb_vx_expr input;
    duckdb_vx_expr lower;
    duckdb_vx_expr upper;
    bool lower_inclusive;
    bool upper_inclusive;
} duckdb_vx_expr_bound_between;

void duckdb_vx_expr_get_bound_between(duckdb_vx_expr expr, duckdb_vx_expr_bound_between *out);

typedef struct {
    duckdb_vx_expr *children;
    size_t children_count;
    duckdb_vx_expr_type type;
} duckdb_vx_expr_bound_operator;

void duckdb_vx_expr_get_bound_operator(duckdb_vx_expr expr, duckdb_vx_expr_bound_operator *out);

typedef struct {
    duckdb_vx_expr *children;
    size_t children_count;
    duckdb_vx_sfunc scalar_function;
    void *bind_info;
} duckdb_vx_expr_bound_function;

void duckdb_vx_expr_get_bound_function(duckdb_vx_expr expr, duckdb_vx_expr_bound_function *out);

#ifdef __cplusplus /* End C ABI */
}
#endif
