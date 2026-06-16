// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb_vx/optimizer.h"
#include "duckdb_vx/table_function.h"
#include "vortex.h"
#include <optional>

/**
 * Our optimizer runs after all duckdb optimizers. Functions that can be pushed
 * down as complex filters (e.g. WHERE str != '') are already pushed down.
 * If there are any functions left, this means they were not pushed down and
 * may produce conflicts (e.g. WHERE prefix("str", 'h')).
 */

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
    FindGetsAndProjections(*plan, analyses, projections);
    if (analyses.empty()) {
        return plan;
    }
    ScalarFnCollect(analyses, projections).VisitOperator(*plan);

    bool any_pushed = false;
    for (auto &[_, analysis] : analyses) {
        for (auto &[column_index, expr] : analysis.col_to_fn) {
            if (expr == nullptr) { // Conflict for column
                continue;
            }
            const TableColumnStorageIndex storage_index =
                analysis.get.GetColumnIds()[column_index].GetPrimaryIndex();
            TableFunctionProjectionExpressionInput input {analysis.get, *expr, storage_index};
            if (projection_expression_pushdown(context, input)) {
                analysis.get.types[column_index] = expr->return_type;
                analysis.get.returned_types[storage_index] = expr->return_type;
                any_pushed = true;
            } else { // failed to push down expression, can't replace it
                expr = nullptr;
            }
        }
    }

    if (any_pushed) {
        ScalarFnReplace(analyses, projections).VisitOperator(*plan);
    }
    return plan;
}

// A passthrough projection only forwards its child columns, e.g. a VIEW's
// "SELECT col".
static bool is_passthrough(const LogicalProjection &projection) {
    if (projection.expressions.empty()) {
        return false; // don't register empty projections in Projections
    }
    for (const auto &e : projection.expressions) {
        if (e->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
            return false;
        }
    }
    return true;
}

void FindGetsAndProjections(LogicalOperator &op, Analyses &analyses, Projections &projections) {
    using enum LogicalOperatorType;
    switch (op.type) {
    case LOGICAL_GET: {
        if (auto &get = op.Cast<LogicalGet>(); get.function.bind == duckdb_vx_table_function_bind) {
            analyses.emplace(get.table_index, GetAnalysis {get, {}});
        }
        break;
    }
    case LOGICAL_PROJECTION: {
        LogicalProjection &projection = op.Cast<LogicalProjection>();
        D_ASSERT(projection.children.size() == 1);
        auto &child = *projection.children[0];
        if (!is_passthrough(projection) || child.type != LOGICAL_GET) {
            break;
        }
        // The GET itself is recorded when recursion reaches it below. Only
        // passthrough projections wrapping a vortex GET act as aliases.
        if (auto &get = child.Cast<LogicalGet>(); get.function.bind == duckdb_vx_table_function_bind) {
            projections.emplace(projection.table_index, projection);
        }
        break;
    }
    default:
        break;
    }

    for (auto &child : op.children) {
        FindGetsAndProjections(*child, analyses, projections);
    }
}

std::optional<GetBinding> Resolve(ColumnBinding binding, Analyses &analyses, const Projections &projections) {
    if (IsVirtualColumn(binding.column_index)) {
        return std::nullopt;
    }
    if (const auto it = analyses.find(binding.table_index); it != analyses.end()) {
        return {{it->second, binding.column_index, nullptr}};
    }

    const auto projection_it = projections.find(binding.table_index);
    if (projection_it == projections.end()) {
        return std::nullopt;
    }

    LogicalProjection &projection = projection_it->second;
    const ExpressionPtr &inner = projection.expressions[binding.column_index];
    if (inner->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::nullopt;
    }
    const ColumnBinding &get_binding = inner->Cast<BoundColumnRefExpression>().binding;
    if (IsVirtualColumn(get_binding.column_index)) {
        return std::nullopt;
    }
    if (const auto it = analyses.find(get_binding.table_index); it != analyses.end()) {
        return {{it->second, get_binding.column_index, &projection}};
    }
    return std::nullopt;
}

void ScalarFnCollect::VisitOperator(LogicalOperator &op) {
    /*
     * Logical projection expressions are columns which reference underlying
     * GETs. Don't process them, as they would add conflicts for every column
     * used in projection. Example: PROJECTION(col) -> GET(col). We don't want
     * to visit BoundColumnRefExpression in PROJECTION to avoid registering a
     * non-existent conflict.
     *
     * However, ScalarFnReplace will visit them because we need to update their
     * types if pushdown succeeded.
     */
    if (op.type == LogicalOperatorType::LOGICAL_PROJECTION &&
        projections.count(op.Cast<LogicalProjection>().table_index)) {
        VisitOperatorChildren(op);
        return;
    }
    LogicalOperatorVisitor::VisitOperator(op);
}

ExpressionPtr ScalarFnCollect::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    if (const auto binding = Resolve(expr.binding, analyses, projections)) {
        // Column is used without function applied to it, register a conflict.
        // Not emplace() as we need to update the value if it was present
        binding->analysis.col_to_fn[binding->column_index] = nullptr;
    }
    return std::move(*ptr);
}

ExpressionPtr ScalarFnCollect::VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) {
    if (expr.children.size() != 1 ||
        expr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        // Descend into children so e.g. fn(col, other) still sees "col" and
        // registers a conflict
        return nullptr;
    }
    const auto &bound_col = expr.children[0]->Cast<BoundColumnRefExpression>();
    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return nullptr;
    }
    auto &col_to_fn = binding->analysis.col_to_fn;

    if (auto it = col_to_fn.find(binding->column_index); it == col_to_fn.end()) {
        // This is the first time we see the column used by a single function.
        col_to_fn.emplace(binding->column_index, &expr);
    } else if (it->second == nullptr || !it->second->Equals(expr)) {
        // Either column is used with different function in "expr" or
        // there already is a conflict.
        it->second = nullptr;
    }

    return std::move(*ptr);
}

static bool can_pushdown_column(const GetAnalysis &analysis, TableColumnScanIndex idx) {
    const auto it = analysis.col_to_fn.find(idx);
    return it != analysis.col_to_fn.end() && it->second != nullptr;
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    const auto binding = Resolve(expr.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }

    const auto &[analysis, column_index, projection] = *binding;
    if (can_pushdown_column(analysis, column_index)) {
        expr.return_type = analysis.get.types[column_index];
        if (projection != nullptr) {
            projection->types[column_index] = expr.return_type;
        }
    }

    return std::move(*ptr);
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) {
    if (expr.children.size() != 1 ||
        expr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return nullptr; // Same as in ScalarFnCollect::VisitReplace
    }
    ExpressionPtr &bound_col_base = expr.children[0];
    const auto &bound_col = bound_col_base->Cast<BoundColumnRefExpression>();
    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return nullptr;
    }

    const auto &[analysis, column_index, projection] = *binding;
    if (!can_pushdown_column(analysis, column_index)) {
        return std::move(*ptr);
    }

    bound_col_base->return_type = analysis.get.types[column_index];
    if (projection != nullptr) {
        projection->types[column_index] = bound_col_base->return_type;
    }
    return std::move(bound_col_base);
}

ScalarFnCollect::ScalarFnCollect(Analyses &analyses, const Projections &projections)
    : analyses(analyses), projections(projections) {
}

ScalarFnReplace::ScalarFnReplace(Analyses &analyses, const Projections &projections)
    : analyses(analyses), projections(projections) {
}
