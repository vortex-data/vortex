// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/expr.h"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"

using namespace duckdb;

extern "C" const char *duckdb_vx_expr_to_string(duckdb_vx_expr ffi_expr) {
    if (!ffi_expr) {
        return nullptr;
    }
    auto expr = reinterpret_cast<Expression *>(ffi_expr);
    auto str = expr->ToString();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

//! Create a DuckDB vortex error.
extern "C" void duckdb_vx_destroy_expr(duckdb_vx_expr *ffi_expr) {
    auto expr = reinterpret_cast<Expression *>(ffi_expr);
    delete expr;
    memset(ffi_expr, 0, sizeof(duckdb_vx_expr));
}

extern "C" duckdb_vx_expr_class duckdb_vx_expr_get_class(duckdb_vx_expr ffi_expr) {
    if (!ffi_expr) {
        return DUCKDB_VX_EXPR_CLASS_INVALID;
    }
    auto expr = reinterpret_cast<Expression *>(ffi_expr);
    return static_cast<duckdb_vx_expr_class>(expr->GetExpressionClass());
}

extern "C" const char *duckdb_vx_expr_get_bound_column_ref_get_name(duckdb_vx_expr ffi_expr) {
    if (!ffi_expr) {
        return nullptr;
    }
    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundColumnRefExpression>();
    auto str = expr.GetName();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

extern "C" duckdb_value duckdb_vx_expr_bound_constant_get_value(duckdb_vx_expr ffi_expr) {
    if (!ffi_expr) {
        return nullptr;
    }
    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundConstantExpression>();
    return reinterpret_cast<duckdb_value>(&expr.value);
}

extern "C" void duckdb_vx_expr_get_bound_comparison(duckdb_vx_expr ffi_expr,
                                                    duckdb_vx_expr_bound_comparison *out) {
    if (!ffi_expr || !out) {
        return;
    }
    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundComparisonExpression>();
    out->left = reinterpret_cast<duckdb_vx_expr>(expr.left.get());
    out->right = reinterpret_cast<duckdb_vx_expr>(expr.right.get());
    out->type = static_cast<duckdb_vx_expr_type>(expr.type);
}

extern "C" void duckdb_vx_expr_get_bound_conjunction(duckdb_vx_expr ffi_expr,
                                                     duckdb_vx_expr_bound_conjunction *out) {
    if (!ffi_expr || !out) {
        return;
    }

    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundConjunctionExpression>();
    out->children_count = expr.children.size();
    out->children = reinterpret_cast<duckdb_vx_expr *>(expr.children.data());
    out->type = static_cast<duckdb_vx_expr_type>(expr.type);
}

extern "C" void duckdb_vx_expr_get_bound_between(duckdb_vx_expr ffi_expr, duckdb_vx_expr_bound_between *out) {
    if (!ffi_expr || !out) {
        return;
    }
    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundBetweenExpression>();
    out->input = reinterpret_cast<duckdb_vx_expr>(expr.input.get());
    out->lower = reinterpret_cast<duckdb_vx_expr>(expr.lower.get());
    out->upper = reinterpret_cast<duckdb_vx_expr>(expr.upper.get());
    out->lower_inclusive = expr.lower_inclusive;
    out->upper_inclusive = expr.upper_inclusive;
}

extern "C" void duckdb_vx_expr_get_bound_operator(duckdb_vx_expr ffi_expr,
                                                  duckdb_vx_expr_bound_operator *out) {
    if (!ffi_expr || !out) {
        return;
    }
    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundOperatorExpression>();
    out->children_count = expr.children.size();
    out->children = reinterpret_cast<duckdb_vx_expr *>(expr.children.data());
    out->type = static_cast<duckdb_vx_expr_type>(expr.type);
}

extern "C" void duckdb_vx_expr_get_bound_function(duckdb_vx_expr ffi_expr,
                                                  duckdb_vx_expr_bound_function *out) {
    if (!ffi_expr || !out) {
        return;
    }
    auto &expr = reinterpret_cast<Expression *>(ffi_expr)->Cast<BoundFunctionExpression>();
    out->children_count = expr.children.size();
    out->children = reinterpret_cast<duckdb_vx_expr *>(expr.children.data());
    out->scalar_function = reinterpret_cast<duckdb_vx_sfunc>(&expr.function);
    out->bind_info = expr.bind_info.get();
}
