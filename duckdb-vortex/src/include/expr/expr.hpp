#pragma once

#include "expr.pb.h"
#include "duckdb/planner/expression.hpp"

#include <duckdb/planner/table_filter.hpp>

vortex::expr::Expr *table_expression_into_expr(google::protobuf::Arena &arena, duckdb::TableFilter &filter,
                                               std::string &column_name);
