// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"

static_assert(sizeof(idx_t) == 8);

// We need this exposed to compare function addresses in optimizer.cpp
duckdb::unique_ptr<duckdb::FunctionData>
duckdb_vx_table_function_bind(duckdb::ClientContext &context,
                              duckdb::TableFunctionBindInput &input,
                              duckdb::vector<duckdb::LogicalType> &return_types,
                              duckdb::vector<duckdb::string> &names);

struct TableFunctionProjectionExpressionInput {
    const duckdb::LogicalGet &get;
    const duckdb::Expression &expression;
    idx_t projection_idx;
};

// true if we can push down the expression, false otherwise
bool projection_expression_pushdown(duckdb::ClientContext &context,
                                    const TableFunctionProjectionExpressionInput &input);
