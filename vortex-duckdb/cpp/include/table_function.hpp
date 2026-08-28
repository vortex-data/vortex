// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"

#include <limits>

using namespace duckdb;

static_assert(sizeof(idx_t) == 8);

bool is_vortex_scan(const TableFunction &function);

constexpr inline idx_t COUNT_STAR_PROJ_IDX = std::numeric_limits<idx_t>::max();

struct TableFunctionUngroupedAggregateInput {
    const LogicalGet &get;
    // Column scan index -> aggregate expression
    const vector<std::pair<idx_t, const Expression &>> &projections;
};

bool aggregate_pushdown(ClientContext &context, const TableFunctionUngroupedAggregateInput &input);
