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

// Only one consumer of this header file, so "using" is fine
using namespace duckdb;

using ExpressionPtr = unique_ptr<Expression>;
using LogicalOperatorPtr = unique_ptr<LogicalOperator>;

// Compatibility with duckdb's main branch where these are structs
using TableColumnIndex = idx_t;
using TableColumnPrimaryIndex = idx_t;

struct GetAnalysis {
    LogicalGet &get;
    /**
     * for fn(col), mapping of "col storage index" -> "fn expression".
     * "fn expression" is nullptr iff column is used with a different function
     * or without function application in the query plan.
     */
    unordered_map<TableColumnPrimaryIndex, const BoundFunctionExpression *> col_to_fn;
};

using Analyses = unordered_map<TableColumnIndex, GetAnalysis>;

/*
 * Query plans may have PROJECTIONs which wrap GETs. One example is VIEWs for
 * our benchmarks:
 *
 * CREATE VIEW view AS (SELECT * FROM '*.vortex');
 * SELECT len(col) FROM view;
 *
 * Second query "col"'s table_index would be 1 (VIEW) and not 0 (GET for
 * vortex). But we want to push down len(col) to vortex. So we keep an aliases
 * mapping of
 *
 * "projection table index" to "projection operator".
 *
 * to resolve this.
 * For simplicity, current implementation is limited to one level i.e.
 * VIEW -> GET is pushed down but VIEW->VIEW->GET or VIEW->CTE->GET is not.
 */
using Projections = unordered_map<TableColumnIndex, const LogicalProjection &>;

/**
 * Collect fn(col) expressions i.e. expressions where a single function (not
 * a function chain) wraps a single bound column. If "col" is used without
 * function application in "plan", record in "analyses.conflicts"
 */
struct ScalarFnCollect final : LogicalOperatorVisitor {
    Analyses &analyses;
    const Projections &projections;

    inline ScalarFnCollect(Analyses &analyses, const Projections &projections)
        : analyses(analyses), projections(projections) {
    }
    void VisitOperator(LogicalOperator &op) override;
    ExpressionPtr VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) override;
    ExpressionPtr VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) override;
};

/*
 * Replace fn(col) without conflicts collected in ScalarFnCollect with "col".
 * Update return types for bound columns as well as logical projections
 * referencing this column.
 */
struct ScalarFnReplace final : LogicalOperatorVisitor {
    Analyses &analyses;
    const Projections &projections;

    inline ScalarFnReplace(Analyses &analyses, const Projections &aliases)
        : analyses(analyses), projections(aliases) {
    }
    ExpressionPtr VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) override;
    ExpressionPtr VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) override;
};

void FindGetsAndAliases(LogicalOperator &op,
                        Analyses &analyses,
                        Projections &aliases,
                        LogicalOperator *parent = nullptr);

LogicalOperatorPtr TryPushdownScalarFunctions(ClientContext &context, LogicalOperatorPtr plan);
void VortexOptimizeFunction(OptimizerExtensionInput &input, LogicalOperatorPtr &plan);

struct VortexOptimizerExtension final : OptimizerExtension {
    inline VortexOptimizerExtension() : OptimizerExtension(VortexOptimizeFunction, nullptr, {}) {
    }
};
#endif
