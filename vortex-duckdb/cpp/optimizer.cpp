// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "optimizer.hpp"
#include "table_function.hpp"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
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
        if (!IsPassthrough(projection)) {
            break;
        }
        LogicalGet *get = nullptr;

        // queries with LIMIT may include a STREAMING_LIMIT between PROJECTION and GET.
        // See sqllogictest scalar_function_pushdown_limit.test
        if (child.type == LOGICAL_LIMIT) {
            if (auto &limit = child.Cast<LogicalLimit>();
                limit.children.size() == 1 && limit.children[0]->type == LOGICAL_GET) {
                get = &limit.children[0]->Cast<LogicalGet>();
            }
        } else if (child.type == LOGICAL_GET) {
            get = &child.Cast<LogicalGet>();
        }

        if (get != nullptr && get->function.bind == duckdb_vx_table_function_bind) {
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

static bool IsVirtualColumn(const GetAnalysis &analysis, TableColumnScanIndex idx) {
    return analysis.get.GetColumnIds()[idx].IsVirtualColumn();
}

std::optional<GetBinding> Resolve(ColumnBinding binding, Analyses &analyses, const Projections &projections) {
    if (const auto it = analyses.find(binding.table_index); it != analyses.end()) {
        if (IsVirtualColumn(it->second, binding.column_index)) {
            return std::nullopt;
        }
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
    if (const auto it = analyses.find(get_binding.table_index); it != analyses.end()) {
        if (IsVirtualColumn(it->second, get_binding.column_index)) {
            return std::nullopt;
        }
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
