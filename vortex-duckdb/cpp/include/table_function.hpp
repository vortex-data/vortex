// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"

using namespace duckdb;

static_assert(sizeof(idx_t) == 8);

bool is_vortex_scan(const TableFunction &function);

struct TableFunctionProjectionExpressionInput {
    const LogicalGet &get;
    const Expression &expression;
    idx_t projection_idx;
};

// true if we can push down the expression, false otherwise
bool projection_expression_pushdown(ClientContext &context,
                                    const TableFunctionProjectionExpressionInput &input);

struct TableFunctionUngroupedAggregateInput {
    const LogicalGet &get;
    // Column scan index -> aggregate expression
    const vector<std::pair<idx_t, const Expression &>> &projections;
};

bool aggregate_pushdown(ClientContext &context, const TableFunctionUngroupedAggregateInput &input);

