// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "data.hpp"
#include "duckdb.h"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"

using namespace duckdb;

static_assert(sizeof(idx_t) == 8);

// We need this exposed to compare function addresses in optimizer.cpp
unique_ptr<FunctionData> duckdb_vx_table_function_bind(ClientContext &context,
                                                       TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types,
                                                       vector<string> &names);

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

struct VortexBindData final : FunctionData {
    VortexBindData(unique_ptr<CData> ffi_data, const vector<LogicalType> &types)
        : ffi_data(std::move(ffi_data)), types(types) {
    }
    unique_ptr<FunctionData> Copy() const override;
    bool Equals(const FunctionData &other) const override;

    unique_ptr<CData> ffi_data;
    vector<LogicalType> types;
};

struct VortexGlobalData final : GlobalTableFunctionState {
    explicit VortexGlobalData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }

    idx_t MaxThreads() const override {
        return GlobalTableFunctionState::MAX_THREADS;
    }

    unique_ptr<CData> ffi_data;
};

struct VortexLocalData final : LocalTableFunctionState {
    explicit VortexLocalData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }
    unique_ptr<CData> ffi_data;
};

struct VortexBindResults {
    vector<LogicalType> &return_types;
    vector<string> &names;
};
