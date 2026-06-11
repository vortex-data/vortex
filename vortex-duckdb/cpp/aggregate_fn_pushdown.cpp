// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "aggregate_fn_pushdown.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "scalar_fn_pushdown.hpp"
#include "table_function.hpp"

/**
 * Mirrors the scalar function pass, but for UNGROUPED_AGGREGATE nodes. We track
 * AGGREGATE -> PROJECTION -> GET triples (the projection is optional) and push
 * each agg(col) into the vortex table function. On success the GET returns one
 * pre-aggregated row, one column per aggregate, so the aggregate node is
 * redundant and we splice the GET in its place.
 *
 * pushdown_projection_aggregates (table_function.rs) duplicates a column field
 * per aggregate in select-list order, so we collect the aggregates in that same
 * order and renumber the GET's columns to 0..N.
 */

using enum LogicalOperatorType;

LogicalOperatorPtr TryPushdownAggregateFunctions(ClientContext &context, LogicalOperatorPtr plan) {
    Analyses analyses;
    Projections projections;
    FindGetsAndProjections(*plan, analyses, projections);
    if (analyses.empty()) {
        return plan;
    }
    return RewriteAggregates(context, std::move(plan), analyses, projections);
}

LogicalOperatorPtr RewriteAggregates(ClientContext &context,
                                     LogicalOperatorPtr op,
                                     Analyses &analyses,
                                     const Projections &projections) {
    for (auto &child : op->children) {
        child = RewriteAggregates(context, std::move(child), analyses, projections);
    }
    if (op->type == LOGICAL_AGGREGATE_AND_GROUP_BY) {
        return TryReplaceAggregate(context, std::move(op), analyses, projections);
    }
    return op;
}

static bool IsUngrouped(const LogicalAggregate &agg) {
    return agg.groups.empty() && agg.grouping_sets.empty() && agg.grouping_functions.empty() &&
           !agg.expressions.empty();
}

// Move GET from UNGROUPED_AGGREGATE -> [GET] or
// UNGROUPED_AGGREGATE -> PROJECTION -> [GET]
static LogicalOperatorPtr MoveGet(LogicalAggregate &agg) {
    auto &child = agg.children[0];
    if (child->type == LOGICAL_GET) {
        return std::move(child);
    }
    D_ASSERT(child->type == LOGICAL_PROJECTION);
    D_ASSERT(child->children.size() == 1);
    D_ASSERT(child->children[0]->type == LOGICAL_GET);
    return std::move(child->children[0]);
}

LogicalOperatorPtr TryReplaceAggregate(ClientContext &context,
                                       LogicalOperatorPtr op,
                                       Analyses &analyses,
                                       const Projections &projections) {
    LogicalAggregate &agg = op->Cast<LogicalAggregate>();
    if (!IsUngrouped(agg)) {
        return op;
    }

    LogicalGet *get = GetChildGet(agg);
    if (get == nullptr) {
        return op;
    }

    // Each aggregate must be a plain agg(col) over this GET. We only support the
    // simple case; distinct/filter/order by or anything other than a single
    // column argument bails out (and pushdown is all-or-nothing per node).
    vector<std::pair<TableColumnStorageIndex, const Expression &>> input;
    const idx_t N = agg.expressions.size();
    input.reserve(N);

    for (const auto &expr : agg.expressions) {
        if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
            return op;
        }
        const auto &bound_aggr = expr->Cast<BoundAggregateExpression>();
        if (bound_aggr.IsDistinct() || bound_aggr.filter != nullptr || bound_aggr.order_bys != nullptr ||
            bound_aggr.children.size() != 1 ||
            bound_aggr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
            return op;
        }
        const auto &bound_col = bound_aggr.children[0]->Cast<BoundColumnRefExpression>();
        const auto binding = Resolve(bound_col.binding, analyses, projections);
        if (!binding || &binding->analysis.get != get) {
            return op;
        }
        const TableColumnStorageIndex storage_index = binding->analysis.StorageIndex(binding->column_index);
        input.emplace_back(storage_index, *expr);
    }

    if (!aggregate_pushdown(context, {*get, input})) {
        return op;
    }

    // The GET now returns one row with one column per aggregate. Renumber its
    // columns to 0..N, take on the aggregate's return types, and adopt the
    // aggregate's table index so the (aggregate_index, k) references upstream
    // keep resolving to column k.
    auto &column_ids = get->GetMutableColumnIds();
    get->types.resize(N);
    get->returned_types.resize(N);
    column_ids.resize(N);

    vector<string> names(N);

    for (idx_t i = 0; i < N; i++) {
        const auto &[storage_index, expr] = input[i];
        names[i] = get->names[storage_index];
        get->types[i] = expr.return_type;
        get->returned_types[i] = expr.return_type;
        column_ids[i] = ColumnIndex {i};
    }
    get->names = std::move(names);
    // TODO projection_ids = column ids?
    get->projection_ids.clear();
    get->table_index = agg.aggregate_index;

    return MoveGet(agg);
}

LogicalGet *GetChildGet(const LogicalAggregate &agg) {
    if (agg.children.size() != 1) {
        return nullptr;
    }
    LogicalOperator &child = *agg.children[0];
    LogicalOperator *get_op;
    if (child.type == LOGICAL_GET) {
        get_op = &child;
    } else if (child.type == LOGICAL_PROJECTION && child.children.size() == 1 &&
               child.children[0]->type == LOGICAL_GET) {
        get_op = child.children[0].get();
    } else {
        return nullptr;
    }
    auto &get = get_op->Cast<LogicalGet>();
    return get.function.bind == duckdb_vx_table_function_bind ? &get : nullptr;
}
