// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "optimizer.hpp"
#include "table_function.hpp"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

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
        if (!IsPassthrough(projection) || child.type != LOGICAL_GET) {
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

TableColumnStorageIndex GetAnalysis::StorageIndex(TableColumnScanIndex idx) const {
    return get.GetColumnIds()[idx].GetPrimaryIndex();
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

bool CanPushdownColumn(const GetAnalysis &analysis, TableColumnScanIndex idx) {
    const auto it = analysis.col_to_expr.find(idx);
    return it != analysis.col_to_expr.end() && it->second != nullptr;
}

bool IsPassthrough(const LogicalProjection &projection) {
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
