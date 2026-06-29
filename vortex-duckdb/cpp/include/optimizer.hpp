// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "table_function.hpp"

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include <optional>

// Aliases here are for ease of migration to duckdb 2.0 where these are
// separate types

using namespace duckdb;

/**
 * Column index in requested scan. Example:
 *
 * CREATE TABLE t (a1 INTEGER, a2 INTEGER, a3 INTEGER);
 * SELECT a2, a3 FROM t;
 *
 * a2's TableColumnScanIndex is 0, a3's TableColumnScanIndex is 1,
 * index is index in SELECT clause.
 */
using TableColumnScanIndex = idx_t;
using ProjectionIndex = TableColumnScanIndex;

/**
 * Column index in table's storage. Example:
 *
 * CREATE TABLE t (a1 INTEGER, a2 INTEGER, a3 INTEGER);
 * SELECT a2, a3 FROM t;
 *
 * a2's TableColumnStorageIndex is 1, a3's TableColumnStorageIndex is 2,
 * index is index of column in table storage.
 *
 * for i: TableColumnScanIndex, column_ids[i].GetPrimaryIndex() is
 * TableColumnStorageIndex
 */
using TableColumnStorageIndex = idx_t;

using TableIndex = idx_t;

using ExpressionPtr = unique_ptr<Expression>;
using LogicalOperatorPtr = unique_ptr<LogicalOperator>;

struct GetAnalysis {
    LogicalGet &get;
    /**
     * for fn(col), mapping of "col scan index" -> "expression applied to function".
     * "expression" is nullptr iff column is used with a different expression
     * or without expression application in the query plan (i.e. SELECT col).
     */
    unordered_map<TableColumnScanIndex, const Expression *> col_to_expr;

    TableColumnStorageIndex StorageIndex(TableColumnScanIndex idx) const;
};

using Analyses = unordered_map<TableIndex, GetAnalysis>;

/*
 * Using scalar function pushdown as a specific example,
 * SELECT fn(col) FROM '*.vortex' yields a PROJECTION fn(col) -> GET (vortex)
 * plan. PROJECTION's "col" table_index is 1, vortex GET's table_index is 0.
 * So we want to track original table_index for GET in case column is found
 * in filter we failed to push down (i.e. WHERE prefix(col, 'h')) as well as
 * projection's table_index.
 *
 * So we keep a mapping of
 *
 * "projection table index" to "projection operator".
 *
 * to resolve this.
 * For simplicity, current implementation is limited to one level i.e.
 * PROJECTION -> GET (i.e. read from VIEW) is pushed down but VIEW->VIEW->GET
 * or VIEW->CTE->GET is not.
 *
 * Storing a reference is fine because the plan outlives the optimizer pass.
 */
using Projections = unordered_map<TableIndex, LogicalProjection &>;

void FindGetsAndProjections(LogicalOperator &op, Analyses &analyses, Projections &aliases);

struct GetBinding {
    GetAnalysis &analysis;
    TableColumnScanIndex column_index;
    // If column binding was part of a projection, this is non-nullptr
    LogicalProjection *projection;
};

/*
 * Given a column binding, resolve it to a GET and a GET's column scan index.
 * Returns nullopt for virtual columns and columns which are neither part of
 * GET nor part of PROJECTION wrapping a GET.
 */
std::optional<GetBinding> Resolve(ColumnBinding binding, Analyses &analyses, const Projections &projections);

// A passthrough projection only forwards its child columns, e.g. a VIEW's
// "SELECT col".
bool IsPassthrough(const LogicalProjection &projection);

// There are no conflicting column usages in the plan
bool CanPushdownColumn(const GetAnalysis &analysis, TableColumnScanIndex idx);

template <class Collect, class Replace>
LogicalOperatorPtr TryPushdown(ClientContext &context, LogicalOperatorPtr plan) {
    Analyses analyses;
    Projections projections;
    FindGetsAndProjections(*plan, analyses, projections);
    if (analyses.empty()) {
        return plan;
    }
    Collect(analyses, projections).VisitOperator(*plan);

    bool any_pushed = false;
    for (auto &[_, analysis] : analyses) {
        for (auto &[column_index, expr] : analysis.col_to_expr) {
            if (expr == nullptr) { // Conflict for column
                continue;
            }
            const TableColumnStorageIndex storage_index = analysis.StorageIndex(column_index);
            TableFunctionProjectionExpressionInput input {analysis.get, *expr, storage_index};
            if (projection_expression_pushdown(context, input)) {
                // LOGICAL_GET doesn't initialize .types of LogicalOperator
                analysis.get.returned_types[storage_index] = expr->return_type;
                any_pushed = true;
            } else { // failed to push down expression, can't replace it
                expr = nullptr;
            }
        }
    }

    if (any_pushed) {
        Replace(analyses, projections).VisitOperator(*plan);
    }
    return plan;
}
