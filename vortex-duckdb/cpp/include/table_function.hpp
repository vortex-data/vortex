// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "data.hpp"
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

/*
 * DuckDB uses partition row groups for two purposes:
 *
 * 1. If optimizer proves query (e.g. SELECT min(col)) can be answered from
 *    metadata, it replaces a real scan with a call to GetColumnStatistics.
 * 2. If optimizer proves a row group can be pruned because of statistics, it
 *    removes row group's read. We don't use this as we implement own prunung.
 *
 * For (1) we care about providing statistics fast, so we report a file as
 * a "row group".
 */
struct VortexRowGroup final : PartitionRowGroup {
    explicit VortexRowGroup(unique_ptr<CData> ffi_footer) : ffi_footer(std::move(ffi_footer)) {
    }

    unique_ptr<CData> ffi_footer;

    unique_ptr<BaseStatistics> GetColumnStatistics(const StorageIndex &storage_index) override;
    bool MinMaxIsExact(const BaseStatistics &, const StorageIndex &) override {
        // TODO(myrrc): in duckdb 2.0 we should report false for strings and
        // also add TRUNCATED_STATS type for them
        return true;
    }
};
