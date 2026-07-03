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
#include "duckdb/common/error_data.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/transaction/meta_transaction.hpp"

#include <exception>

using namespace duckdb;

extern "C" const char *duckdb_vx_sfunc_name(duckdb_vx_sfunc ffi_func) {
    if (!ffi_func) {
        return nullptr;
    }
    auto func = reinterpret_cast<ScalarFunction *>(ffi_func);
    return func->name.c_str();
}

extern "C" duckdb_state duckdb_vx_register_st_dwithin_override(duckdb_database ffi_db) {
    if (!ffi_db) {
        return DuckDBError;
    }
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;
    try {
        Connection conn(db);
        ClientContext &context = *conn.context;
        context.RunFunctionInTransaction([&]() {
            auto &system = Catalog::GetSystemCatalog(context);
            auto entry = system.GetEntry<ScalarFunctionCatalogEntry>(context,
                                                                     DEFAULT_SCHEMA,
                                                                     "st_dwithin",
                                                                     OnEntryNotFound::RETURN_NULL);
            if (!entry) {
                // No `spatial` loaded, so there is no `ST_DWithin` to override.
                return;
            }
            ScalarFunctionSet set("st_dwithin");
            for (const auto &overload : entry->functions.functions) {
                ScalarFunction copy = overload;
                // Keep the radius as children[2]; spatial's bind folds it into private bind data.
                copy.bind = nullptr;
                set.AddFunction(copy);
            }
            CreateScalarFunctionInfo info(std::move(set));
            info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
            // `internal` entries are only accepted by the system catalog.
            info.internal = false;
            // The user catalog binds ahead of the system catalog, shadowing spatial's entry;
            // `RestoreStDWithin` rebinds unpushed calls through the original.
            auto &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
            // Durable catalogs require the modified mark; scalar function entries are never
            // persisted, so this is metadata-only.
            MetaTransaction::Get(context).ModifyDatabase(catalog.GetAttached(), DatabaseModificationType());
            catalog.CreateFunction(context, info);
        });
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to register the ST_DWithin override:\t" + data.Message());
        return DuckDBError;
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

extern "C" duckdb_logical_type duckdb_vx_expr_get_return_type(duckdb_vx_expr ffi_expr) {
    D_ASSERT(ffi_expr);
    auto expr = reinterpret_cast<Expression *>(ffi_expr);
    return reinterpret_cast<duckdb_logical_type>(&expr->return_type);
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
