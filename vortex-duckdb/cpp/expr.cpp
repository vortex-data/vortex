// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
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

// Get detailed debug string representation of expression
extern "C" char *duckdb_vx_expr_to_debug_string(duckdb_vx_expr ffi_expr) {
    try {
        if (!ffi_expr) {
            return nullptr;
        }

        auto expr = reinterpret_cast<Expression *>(ffi_expr);

        // Create detailed debug string with class, type, and content information
        std::string debug_str = "Expression Debug Info:\n";
        debug_str += "  Class: " + ExpressionClassToString(expr->GetExpressionClass()) + "\n";
        debug_str += "  Type: " + ExpressionTypeToString(expr->GetExpressionType()) + "\n";
        debug_str += "  Return Type: " + expr->return_type.ToString() + "\n";
        debug_str += "  ToString(): " + expr->ToString() + "\n";

        // Add specific information based on expression class
        switch (expr->GetExpressionClass()) {
        case ExpressionClass::BOUND_COLUMN_REF: {
            auto &col_ref = expr->Cast<BoundColumnRefExpression>();
            debug_str += "  Column Binding: table=" + std::to_string(col_ref.binding.table_index) +
                         ", column=" + std::to_string(col_ref.binding.column_index) + "\n";
            debug_str += "  Depth: " + std::to_string(col_ref.depth) + "\n";
            break;
        }
        case ExpressionClass::BOUND_FUNCTION: {
            auto &func_expr = expr->Cast<BoundFunctionExpression>();
            debug_str += "  Function: " + func_expr.function.name + "\n";
            debug_str += "  Arguments: " + std::to_string(func_expr.children.size()) + "\n";
            for (size_t i = 0; i < func_expr.children.size(); i++) {
                debug_str += "    [" + std::to_string(i) + "] " + func_expr.children[i]->ToString() + "\n";
            }
            break;
        }
        case ExpressionClass::BOUND_CONSTANT: {
            auto &const_expr = expr->Cast<BoundConstantExpression>();
            debug_str += "  Value: " + const_expr.value.ToString() + "\n";
            break;
        }
        case ExpressionClass::BOUND_COMPARISON: {
            auto &comp_expr = expr->Cast<BoundComparisonExpression>();
            debug_str += "  Left: " + comp_expr.left->ToString() + "\n";
            debug_str += "  Right: " + comp_expr.right->ToString() + "\n";
            break;
        }
        case ExpressionClass::BOUND_CONJUNCTION: {
            auto &conj_expr = expr->Cast<BoundConjunctionExpression>();
            debug_str += "  Children: " + std::to_string(conj_expr.children.size()) + "\n";
            for (size_t i = 0; i < conj_expr.children.size(); i++) {
                debug_str += "    [" + std::to_string(i) + "] " + conj_expr.children[i]->ToString() + "\n";
            }
            break;
        }
        case ExpressionClass::BOUND_OPERATOR: {
            auto &op_expr = expr->Cast<BoundOperatorExpression>();
            debug_str += "  Children: " + std::to_string(op_expr.children.size()) + "\n";
            for (size_t i = 0; i < op_expr.children.size(); i++) {
                debug_str += "    [" + std::to_string(i) + "] " + op_expr.children[i]->ToString() + "\n";
            }
            break;
        }
        default:
            debug_str += "  (No additional debug info for this expression class)\n";
            break;
        }

        // Allocate C string and copy
        char *result = static_cast<char *>(malloc(debug_str.length() + 1));
        if (result) {
            strcpy(result, debug_str.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Legacy alias for backwards compatibility with optimizer_rule.h
extern "C" char *duckdb_vx_expression_to_string(duckdb_vx_expr ffi_expr) {
    return const_cast<char *>(duckdb_vx_expr_to_string(ffi_expr));
}

//! Create a DuckDB vortex error.
extern "C" void duckdb_vx_destroy_expr(duckdb_vx_expr *ffi_expr) {
    if (ffi_expr == nullptr) {
        return;
    }
    auto expr = reinterpret_cast<Expression *>(*ffi_expr);
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
