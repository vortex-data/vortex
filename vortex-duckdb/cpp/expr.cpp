// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "expr.h"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include <exception>

using namespace duckdb;

extern "C" const char *duckdb_vx_sfunc_name(duckdb_vx_sfunc ffi_func) {
    if (!ffi_func) {
        return nullptr;
    }
    auto func = reinterpret_cast<ScalarFunction *>(ffi_func);
    return func->name.c_str();
}

extern "C" duckdb_state duckdb_vx_register_geo_aliases(duckdb_database ffi_db) {
    if (!ffi_db) {
        return DuckDBError;
    }
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    try {
        Connection conn(*wrapper.database->instance);
        ClientContext &context = *conn.context;
        context.RunFunctionInTransaction([&]() {
            auto &catalog = Catalog::GetSystemCatalog(context);
            auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
                context, DEFAULT_SCHEMA, "st_dwithin");
            // Copy each ST_DWithin overload to a non-throwing `vortex_dwithin` so DuckDB will push it.
            ScalarFunctionSet set("vortex_dwithin");
            for (const auto &overload : entry.functions.functions) {
                ScalarFunction copy = overload;
                copy.name = "vortex_dwithin";
                copy.SetErrorMode(FunctionErrors::CANNOT_ERROR);
                // Clear the bind so the radius stays as children[2] for the Vortex converter
                // (ST_DWithin's bind folds it into bind_data). vortex_dwithin is only pushed, never run.
                copy.bind = nullptr;
                set.AddFunction(copy);
            }
            CreateScalarFunctionInfo info(std::move(set));
            info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
            catalog.CreateFunction(context, info);
        });
    } catch (const std::exception &) {
        // No `spatial` loaded, so there is no `ST_DWithin` to alias; nothing to register.
        return DuckDBSuccess;
    }
    return DuckDBSuccess;
}

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
