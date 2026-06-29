// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "scalar_fn_pushdown.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include <optional>

/**
 * Our optimizer runs after all duckdb optimizers. Functions that can be pushed
 * down as complex filters (e.g. WHERE str != '') are already pushed down.
 * If there are any functions left, this means they were not pushed down and
 * may produce conflicts (e.g. WHERE prefix("str", 'h')).
 */
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
        binding->analysis.col_to_expr[binding->column_index] = nullptr;
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
    auto &col_to_expr = binding->analysis.col_to_expr;

    if (auto it = col_to_expr.find(binding->column_index); it == col_to_expr.end()) {
        // This is the first time we see the column used by a single function.
        col_to_expr.emplace(binding->column_index, &expr);
    } else if (it->second == nullptr || !it->second->Equals(expr)) {
        // Either column is used with different function in "expr" or
        // there already is a conflict.
        it->second = nullptr;
    }

    return std::move(*ptr);
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    const auto binding = Resolve(expr.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }

    const auto &[analysis, column_index, projection] = *binding;
    if (CanPushdownColumn(analysis, column_index)) {
        const idx_t storage_index = analysis.get.GetColumnIds()[column_index].GetPrimaryIndex();
        const LogicalType return_type = analysis.get.returned_types[storage_index];
        expr.return_type = return_type;
        if (projection != nullptr && !projection->types.empty()) {
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
    if (!CanPushdownColumn(analysis, column_index)) {
        return std::move(*ptr);
    }

    const idx_t storage_index = analysis.get.GetColumnIds()[column_index].GetPrimaryIndex();
    const LogicalType return_type = analysis.get.returned_types[storage_index];
    bound_col_base->return_type = return_type;
    if (projection != nullptr && !projection->types.empty()) {
        projection->types[column_index] = return_type;
    }
    return std::move(bound_col_base);
}

ScalarFnCollect::ScalarFnCollect(Analyses &analyses, const Projections &projections)
    : analyses(analyses), projections(projections) {
}

ScalarFnReplace::ScalarFnReplace(Analyses &analyses, const Projections &projections)
    : analyses(analyses), projections(projections) {
}
