// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

using namespace duckdb;

/*
 * Vortex shadows some spatial (`ST_*`) scalar functions with tweaked copies, because as
 * spatial registers them their filters can never push into Vortex scans.
 *
 * `duckdb_vx_register_spatial_overrides` (spatial_overrides.h) installs the copies after
 * `LOAD spatial`. `RestoreSpatialOverrides` below hands join conditions back to spatial's
 * originals: Vortex never pushes join conditions, and spatial's join machinery expects its
 * own functions.
 */

// Rebinds overridden spatial calls in join conditions back to spatial's original, so spatial's
// own machinery handles joins. Filters are left untouched: they keep the override and push to
// Vortex scans.
struct SpatialOverrideRestore final : LogicalOperatorVisitor {
    ClientContext &context;

    explicit SpatialOverrideRestore(ClientContext &context);
    void VisitOperator(LogicalOperator &op) override;
    unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *ptr) override;
};

/// Rebind overridden spatial calls in join conditions to spatial's originals.
void RestoreSpatialOverrides(ClientContext &context, LogicalOperator &plan);
