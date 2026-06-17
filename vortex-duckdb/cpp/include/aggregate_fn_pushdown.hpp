// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "duckdb/optimizer/optimizer_extension.hpp"

using namespace duckdb;

using ExpressionPtr = unique_ptr<Expression>;
using LogicalOperatorPtr = unique_ptr<LogicalOperator>;

LogicalOperatorPtr TryPushdownAggregateFunctions(ClientContext &context, LogicalOperatorPtr plan);
