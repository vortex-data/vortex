// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb_vx/optimizer.h"
#include "duckdb_vx/table_function.h"
#include "vortex.h"
#include <optional>

extern "C" duckdb_state duckdb_vx_optimizer_extension_register(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;
    try {
        DBConfig::GetConfig(db).GetCallbackManager().Register(VortexOptimizerExtension());
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex optimizer extension:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}

void VortexOptimizeFunction(OptimizerExtensionInput &input, LogicalOperatorPtr &plan) {
    plan = TryPushdownScalarFunctions(input.context, std::move(plan));
}

LogicalOperatorPtr TryPushdownScalarFunctions(ClientContext &context, LogicalOperatorPtr plan) {
    Analyses analyses;
    Projections projections;
    FindGetsAndAliases(*plan, analyses, projections);
    if (analyses.empty()) {
        return plan;
    }

    ScalarFnCollect(analyses, projections).VisitOperator(*plan);

    bool any_pushed = false;
    for (auto &[_, analysis] : analyses) {
        for (const auto &[storage_idx, expr] : analysis.col_to_fn) {
            if (expr == nullptr) { // Conflict for column
                continue;
            }
            TableFunctionProjectionExpressionInput input {analysis.get, *expr, storage_idx};
            if (projection_expression_pushdown(context, input)) {
                analysis.get.returned_types[storage_idx] = expr->return_type;
                any_pushed = true;
            }
        }
    }

    if (any_pushed) {
        ScalarFnReplace(analyses, projections).VisitOperator(*plan);
    }
    return plan;
}

void FindGetsAndAliases(LogicalOperator &op,
                        Analyses &analyses,
                        Projections &projections,
                        LogicalOperator *parent) {
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &get = op.Cast<LogicalGet>();
        if (get.function.bind == bind) {
            analyses.emplace(get.table_index, GetAnalysis {get, {}});
            if (parent && parent->type == LogicalOperatorType::LOGICAL_PROJECTION) {
                const auto &projection = parent->Cast<LogicalProjection>();
                projections.emplace(projection.table_index, projection);
            }
        }
    }
    for (auto &child : op.children) {
        FindGetsAndAliases(*child, analyses, projections, &op);
    }
}

// Resolve column index to it primary index (storage index).
static TableColumnPrimaryIndex columnPrimaryIndex(const GetAnalysis &analysis, TableColumnIndex idx) {
    return analysis.get.GetColumnIds()[idx].GetPrimaryIndex();
}

/*
 * For a given (table index, column index) pair, resolve it to a GET and a
 * GET's column storage index.
 * Returns nullopt for virtual columns and columns which are neither part of
 * GET nor part of PROJECTION wrapping a GET.
 */
static std::optional<std::pair<GetAnalysis &, TableColumnPrimaryIndex>>
Resolve(ColumnBinding binding, Analyses &analyses, const Projections &projections) {
    if (IsVirtualColumn(binding.column_index)) {
        return std::nullopt;
    }
    if (const auto it = analyses.find(binding.table_index); it != analyses.end()) {
        return {{it->second, columnPrimaryIndex(it->second, binding.column_index)}};
    }

    const auto projection_it = projections.find(binding.table_index);
    if (projection_it == projections.end()) {
        return std::nullopt;
    }
    const ExpressionPtr &inner = projection_it->second.expressions[binding.column_index];
    if (inner->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::nullopt;
    }
    const auto [table_index, column_index] = inner->Cast<BoundColumnRefExpression>().binding;
    if (IsVirtualColumn(column_index)) {
        return std::nullopt;
    }
    if (const auto it = analyses.find(table_index); it != analyses.end()) {
        return {{it->second, columnPrimaryIndex(it->second, column_index)}};
    }
    return std::nullopt;
}

void ScalarFnCollect::VisitOperator(LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_PROJECTION &&
        projections.count(op.Cast<LogicalProjection>().table_index)) {
        // Logical projection expressions are columns which reference underlying
        // GETs. Don't process them, as they would add conflicts for every
        // column used in projection. Example:
        // PROJECTION(col) -> GET(col). We don't want to visit
        // BoundColumnRefExpression in PROJECTION.
        VisitOperatorChildren(op);
        return;
    }
    LogicalOperatorVisitor::VisitOperator(op);
}

ExpressionPtr ScalarFnCollect::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    if (const auto binding = Resolve(expr.binding, analyses, projections)) {
        auto &[analysis, storage_index] = *binding;
        // Column is used without function applied to it, register as a
        // conflict.
        analysis.col_to_fn[storage_index] = nullptr;
    }
    return std::move(*ptr);
}

ExpressionPtr ScalarFnCollect::VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) {
    if (expr.children.size() != 1 ||
        expr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::move(*ptr);
    }
    const auto &bound_col = expr.children[0]->Cast<BoundColumnRefExpression>();

    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }
    auto &[analysis, storage_idx] = *binding;
    if (auto it = analysis.col_to_fn.find(storage_idx); it == analysis.col_to_fn.end()) {
        // This is the first time we see the column used by a single function.
        analysis.col_to_fn.emplace(storage_idx, &expr);
    } else if (it->second != &expr) {
        // Either column is used with different function in "expr" or
        // it->second is nullptr which indicates an existing conflict.
        it->second = nullptr;
    }

    // We don't want to descend into child BoundColumnRefExpression because we
    // have already registered a conflict if it was present.
    return std::move(*ptr);
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    if (const auto binding = Resolve(expr.binding, analyses, projections)) {
        const auto &[analysis, storage_idx] = *binding;
        // We updated GET's return type in TryPushdownScalarFunctions
        expr.return_type = analysis.get.returned_types[storage_idx];
    }
    return std::move(*ptr);
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) {
    if (expr.children.size() != 1 ||
        expr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::move(*ptr);
    }
    ExpressionPtr &bound_col_base = expr.children[0];
    const auto &bound_col = bound_col_base->Cast<BoundColumnRefExpression>();
    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }
    const auto &[analysis, storage_idx] = *binding;

    // If we resolved this column, ScalarFnCollect has already processed it,
    // and expr_map either holds a valid pointer or a nullptr if there's a
    // conflict.
    D_ASSERT(analysis.col_to_fn.find(storage_idx) != analysis.col_to_fn.end());
    if (analysis.col_to_fn[storage_idx] == nullptr) {
        // This column has a conflict, don't replace it
        return std::move(*ptr);
    }

    // We updated GET's return type in TryPushdownScalarFunctions
    bound_col_base->return_type = analysis.get.returned_types[storage_idx];
    return std::move(bound_col_base);
}
