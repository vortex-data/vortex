// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "aggregate_fn_pushdown.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "optimizer.hpp"
#include "table_function.hpp"

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

constexpr inline idx_t COUNT_STAR_PROJ_IDX = std::numeric_limits<TableColumnStorageIndex>::max();

LogicalOperatorPtr TryReplaceAggregate(ClientContext &context,
                                       LogicalOperatorPtr op,
                                       Analyses &analyses,
                                       const Projections &projections) {
    LogicalAggregate &agg = op->Cast<LogicalAggregate>();
    if (!IsUngrouped(agg)) {
        return op;
    }

    LogicalGet *const get = GetChildGet(agg);
    if (get == nullptr) {
        return op;
    }

    vector<std::pair<TableColumnScanIndex, const Expression &>> input;
    const idx_t aggregates_len = agg.expressions.size();
    input.reserve(aggregates_len);

    for (const auto &expr : agg.expressions) {
        if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
            return op;
        }
        const auto &bound_aggr = expr->Cast<BoundAggregateExpression>();
        if (bound_aggr.IsDistinct() || bound_aggr.filter != nullptr || bound_aggr.order_bys != nullptr) {
            return op;
        }

        if (bound_aggr.function.name == "count_star") {
            input.emplace_back(COUNT_STAR_PROJ_IDX, *expr);
            continue;
        }

        if (bound_aggr.children.size() != 1 ||
            bound_aggr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
            return op;
        }
        const auto &bound_col = bound_aggr.children[0]->Cast<BoundColumnRefExpression>();
        const auto binding = Resolve(bound_col.binding, analyses, projections);
        if (!binding || &binding->analysis.get != get) {
            return op;
        }
        input.emplace_back(binding->column_index, *expr);
    }

    if (!aggregate_pushdown(context, {*get, input})) {
        return op;
    }

    // GET now returns one column per aggregate. Expand existing columns
    auto &column_ids = get->GetMutableColumnIds();
    get->types.resize(aggregates_len);
    get->returned_types.resize(aggregates_len);
    column_ids.resize(aggregates_len);

    vector<string> names(aggregates_len); // need a copy because we reference original names

    for (idx_t i = 0; i < aggregates_len; i++) {
        const auto &[column_index, expr] = input[i];
        if (column_index == COUNT_STAR_PROJ_IDX) {
            names[i] = "count_star()";
        } else {
            const TableColumnStorageIndex storage_index = get->GetColumnIds()[column_index].GetPrimaryIndex();
            names[i] = get->names[storage_index];
        }
        get->types[i] = expr.return_type;
        get->returned_types[i] = expr.return_type;
        column_ids[i] = ColumnIndex {i};
    }
    get->names = std::move(names);
    get->projection_ids.clear();
    get->table_index = agg.aggregate_index;

    unique_ptr<LogicalOperator> &child = agg.children[0];
    if (child->type == LOGICAL_GET) {
        return std::move(child);
    }
    D_ASSERT(child->type == LOGICAL_PROJECTION);
    D_ASSERT(child->children.size() == 1);
    D_ASSERT(child->children[0]->type == LOGICAL_GET);
    return std::move(child->children[0]);
}

LogicalGet *GetChildGet(const LogicalAggregate &agg) {
    if (agg.children.size() != 1) {
        return nullptr;
    }
    LogicalOperator &child = *agg.children[0];
    LogicalOperator *op;
    if (child.type == LOGICAL_GET) {
        op = &child;
    } else if (child.type == LOGICAL_PROJECTION && child.children.size() == 1 &&
               child.children[0]->type == LOGICAL_GET) {
        op = child.children[0].get();
    } else {
        return nullptr;
    }
    LogicalGet &get = op->Cast<LogicalGet>();
    return get.function.bind == duckdb_vx_table_function_bind ? &get : nullptr;
}
