// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "optimizer.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

using namespace duckdb;

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
