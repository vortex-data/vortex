#pragma once

#include "expr.pb.h"
#include "duckdb/planner/expression.hpp"

#include <duckdb/planner/table_filter.hpp>

vortex::expr::Expr *table_expression_into_expr(google::protobuf::Arena &arena, duckdb::TableFilter &filter,
                                               const std::string &column_name);

vortex::expr::Expr *flatten_exprs(google::protobuf::Arena &arena, duckdb::vector<vortex::expr::Expr *> &child_filters);