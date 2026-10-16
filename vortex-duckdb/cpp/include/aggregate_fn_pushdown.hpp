// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/type_pushdown.hpp"

using namespace duckdb;

using LogicalOperatorPtr = unique_ptr<LogicalOperator>;

// Push UNGROUPED_AGGREGATE's of form agg(T) and count_star() into GET.
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
