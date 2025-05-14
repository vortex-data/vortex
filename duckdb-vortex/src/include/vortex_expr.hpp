#pragma once

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "expr.pb.h"

namespace vortex {
vortex::expr::Expr *table_expression_into_expr(google::protobuf::Arena &arena, duckdb::TableFilter &filter,
                                               const std::string &column_name);
vortex::expr::Expr *expression_into_vortex_expr(google::protobuf::Arena &arena, const duckdb::Expression &expr);
vortex::expr::Expr *flatten_exprs(google::protobuf::Arena &arena,
                                  const duckdb::vector<vortex::expr::Expr *> &child_filters);
} // namespace vortex
