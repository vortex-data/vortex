// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "optimizer.hpp"

#include "duckdb/common/unordered_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/logical_operator.hpp"

using namespace duckdb;

/**
 * Collect CAST(col) expressions. If "col" is used without CAST in "plan",
 * record in "analyses.conflicts"
 */
struct CastCollect final : LogicalOperatorVisitor {
    Analyses &analyses;
    const Projections &projections;
    // Casts that are direct outputs of a clean projection over a GET. Only these
    // start a pushdown candidate; a nested cast may push down a different value.
    // See test/sql/copy/csv/auto/test_large_integer_detection.test in duckdb
    unordered_set<const Expression *> top_level_casts;

    CastCollect(Analyses &analyses, const Projections &projections);
    void VisitOperator(LogicalOperator &op) override;
    ExpressionPtr VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) override;
    ExpressionPtr VisitReplace(BoundCastExpression &expr, ExpressionPtr *ptr) override;
};

/*
 * For "col" in columns collected by CastCollect, replace CAST(col) to "col"
 * if "col" doesn't have conflicting usage. Update return types for bound
 * columns and logical projections referencing this column.
 */
struct CastReplace final : LogicalOperatorVisitor {
    Analyses &analyses;
    const Projections &projections;

    CastReplace(Analyses &analyses, const Projections &aliases);
    ExpressionPtr VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) override;
    ExpressionPtr VisitReplace(BoundCastExpression &expr, ExpressionPtr *ptr) override;
};
