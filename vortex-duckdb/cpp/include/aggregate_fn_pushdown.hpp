// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "scalar_fn_pushdown.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

using namespace duckdb;

// Push UNGROUPED_AGGREGATE aggregates of form agg(T) into GET.
LogicalOperatorPtr TryPushdownAggregateFunctions(ClientContext &context, LogicalOperatorPtr plan);

LogicalOperatorPtr RewriteAggregates(ClientContext &context,
                                     LogicalOperatorPtr op,
                                     Analyses &analyses,
                                     const Projections &projections);

LogicalOperatorPtr TryReplaceAggregate(ClientContext &context,
                                       LogicalOperatorPtr op,
                                       Analyses &analyses,
                                       const Projections &projections);

// return GET for UNGROUPED_AGGREGATE -> [GET] or for UNGROUPED_AGGREGATE ->
// PROJECTION -> [GET], nullptr if not found.
LogicalGet *GetChildGet(const LogicalAggregate &agg);
