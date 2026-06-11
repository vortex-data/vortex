// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "duckdb.h"

#ifdef __cplusplus
extern "C" {
#endif

duckdb_state duckdb_vx_optimizer_extension_register(duckdb_database ffi_db);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include <optional>

// Only one consumer of this header file, so "using" is fine
using namespace duckdb;

using ExpressionPtr = unique_ptr<Expression>;
using LogicalOperatorPtr = unique_ptr<LogicalOperator>;

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

/**
 * Column index in table's storage. Example:
 *
 * CREATE TABLE t (a1 INTEGER, a2 INTEGER, a3 INTEGER);
 * SELECT a2, a3 FROM t;
 *
 * a2's TableColumnStorageIndex is 1, a3's TableColumnScanIndex is 2,
 * index is index of column in table storage.
 *
 * for i: TableColumnScanIndex, column_ids[i].GetPrimaryIndex() is
 * TableColumnStorageIndex
 */
using TableColumnStorageIndex = idx_t;

using TableIndex = idx_t;

struct GetAnalysis {
    LogicalGet &get;
    /**
     * for fn(col), mapping of "col scan index" -> "fn expression".
     * "fn expression" is nullptr iff column is used with a different function
     * or without function application in the query plan.
     */
    unordered_map<TableColumnScanIndex, const BoundFunctionExpression *> col_to_fn;
};

using Analyses = unordered_map<TableIndex, GetAnalysis>;

/*
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

/**
 * Collect fn(col) expressions i.e. expressions where a single function (not
 * a function chain) wraps a single bound column. If "col" is used without
 * function application in "plan", record in "analyses.conflicts"
 */
struct ScalarFnCollect final : LogicalOperatorVisitor {
    Analyses &analyses;
    const Projections &projections;

    ScalarFnCollect(Analyses &analyses, const Projections &projections);
    void VisitOperator(LogicalOperator &op) override;
    ExpressionPtr VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) override;
    ExpressionPtr VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) override;
};

/*
 * For "col" in columns collected by ScalarFnCollect, replace fn(col) to "col"
 * if "col" doesn't have conflicting usage. Update return types for bound
 * columns and logical projections referencing this column.
 */
struct ScalarFnReplace final : LogicalOperatorVisitor {
    Analyses &analyses;
    const Projections &projections;

    ScalarFnReplace(Analyses &analyses, const Projections &aliases);
    ExpressionPtr VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) override;
    ExpressionPtr VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) override;
};

void FindGetsAndProjections(LogicalOperator &op, Analyses &analyses, Projections &aliases);

LogicalOperatorPtr TryPushdownScalarFunctions(ClientContext &context, LogicalOperatorPtr plan);
void VortexOptimizeFunction(OptimizerExtensionInput &input, LogicalOperatorPtr &plan);

struct VortexOptimizerExtension final : OptimizerExtension {
    inline VortexOptimizerExtension() : OptimizerExtension(VortexOptimizeFunction, nullptr, {}) {
    }
};

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

#endif
