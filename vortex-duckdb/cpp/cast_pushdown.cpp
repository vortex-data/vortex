// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "cast_pushdown.hpp"
#include "table_function.hpp"

#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

// A GET reachable through a single-child chain of filters/projections. A join
// (or any other multi-child operator) breaks the chain.
// See test/sql/copy/csv/test_insert_into_types.test in duckdb (cast not pushed past a join)
static bool ReachesPushdownGet(const LogicalOperator &op) {
    const LogicalOperator *cur = &op;
    while (cur->children.size() == 1) {
        cur = cur->children[0].get();
        switch (cur->type) {
        case LogicalOperatorType::LOGICAL_GET:
            return cur->Cast<LogicalGet>().function.bind == duckdb_vx_table_function_bind;
        case LogicalOperatorType::LOGICAL_FILTER:
        case LogicalOperatorType::LOGICAL_PROJECTION:
            continue;
        default:
            return false;
        }
    }
    return false;
}

void CastCollect::VisitOperator(LogicalOperator &op) {
    /*
     * Logical projection expressions are columns which reference underlying
     * GETs. Don't process them, as they would add conflicts for every column
     * used in projection. Example: PROJECTION(col) -> GET(col). We don't want
     * to visit BoundColumnRefExpression in PROJECTION to avoid registering a
     * non-existent conflict.
     *
     * However, CastReplace will visit them because we need to update their
     * types if pushdown succeeded.
     */
    if (op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
        return LogicalOperatorVisitor::VisitOperator(op);
    }
    auto &projection = op.Cast<LogicalProjection>();

    // Only push casts from a projection that forwards just column refs and
    // casts and reaches a GET without a join in between. A constant or other
    // expression makes the projection ineligible.
    // See test/sql/copy/csv/test_csv_error_message_type.test (top-level cast
    // to VARCHAR must still push) and test_large_integer_detection.test (a
    // nested cast to VARCHAR must not) in duckdb.
    bool clean = ReachesPushdownGet(projection);
    for (const auto &e : projection.expressions) {
        switch (e->GetExpressionClass()) {
        case ExpressionClass::BOUND_COLUMN_REF:
        case ExpressionClass::BOUND_CAST:
            continue;
        default:
            clean = false;
            break;
        }
    }
    if (clean) {
        for (const auto &e : projection.expressions) {
            if (e->GetExpressionClass() == ExpressionClass::BOUND_CAST) {
                top_level_casts.insert(e.get());
            }
        }
    }
    if (projections.count(projection.table_index)) {
        VisitOperatorChildren(op);
        return;
    }

    LogicalOperatorVisitor::VisitOperator(op);
}

ExpressionPtr CastCollect::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    if (const auto binding = Resolve(expr.binding, analyses, projections)) {
        // Column is used without cast applied to it, register a conflict.
        // Not emplace() as we need to update the value if it was present
        binding->analysis.col_to_expr[binding->column_index] = nullptr;
    }
    return std::move(*ptr);
}

ExpressionPtr CastCollect::VisitReplace(BoundCastExpression &expr, ExpressionPtr *ptr) {
    if (expr.child->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        // Descend into children so e.g. fn(col, other) still sees "col" and
        // registers a conflict
        return nullptr;
    }
    const auto &bound_col = expr.child->Cast<BoundColumnRefExpression>();
    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return nullptr;
    }
    auto &col_to_expr = binding->analysis.col_to_expr;

    if (auto it = col_to_expr.find(binding->column_index); it == col_to_expr.end()) {
        // Only a top-level projection cast starts a candidate.
        if (top_level_casts.count(&expr)) {
            col_to_expr.emplace(binding->column_index, &expr);
        }
    } else if (it->second == nullptr ||
               it->second->Cast<BoundCastExpression>().return_type != expr.return_type ||
               // TODO(myrrc) this line needs upstreaming
               it->second->Cast<BoundCastExpression>().try_cast != expr.try_cast) {
        // Different target type, or already a conflict.
        it->second = nullptr;
    }

    return std::move(*ptr);
}

ExpressionPtr CastReplace::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    const auto binding = Resolve(expr.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }

    const auto &[analysis, column_index, projection] = *binding;
    if (CanPushdownColumn(analysis, column_index)) {
        const idx_t storage_index = analysis.get.GetColumnIds()[column_index].GetPrimaryIndex();
        const LogicalType return_type = analysis.get.returned_types[storage_index];
        expr.return_type = return_type;
        // LogicalProjection types are resolved by calling
        // LogicalProjection::ResolveTypes, so we need to check whether types in
        // projection have been resolved, and updated them only if needed.
        if (projection != nullptr && !projection->types.empty()) {
            projection->types[column_index] = return_type;
        }
    }

    return std::move(*ptr);
}

ExpressionPtr CastReplace::VisitReplace(BoundCastExpression &expr, ExpressionPtr *ptr) {
    if (expr.child->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return nullptr; // Same as in ScalarFnCollect::VisitReplace
    }
    auto &bound_col_base = expr.child;
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
    // Same as in CastReplace::VisitReplace(BoundColumnRefExpression)
    if (projection != nullptr && !projection->types.empty()) {
        projection->types[column_index] = return_type;
    }
    return std::move(bound_col_base);
}

CastCollect::CastCollect(Analyses &analyses, const Projections &projections)
    : analyses(analyses), projections(projections) {
}

CastReplace::CastReplace(Analyses &analyses, const Projections &projections)
    : analyses(analyses), projections(projections) {
}
