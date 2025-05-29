#pragma once

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "expr.pb.h"

namespace vortex {
// vortex expr proto ids.
const std::string BETWEEN_ID = "between";
const std::string BINARY_ID = "binary";
const std::string GET_ITEM_ID = "get_item";
const std::string IDENTITY_ID = "identity";
const std::string LIKE_ID = "like";
const std::string LITERAL_ID = "literal";
const std::string NOT_ID = "not";

vortex::expr::Expr *table_expression_into_expr(google::protobuf::Arena &arena, duckdb::TableFilter &filter,
                                               const std::string &column_name);
vortex::expr::Expr *expression_into_vortex_expr(google::protobuf::Arena &arena, const duckdb::Expression &expr);
vortex::expr::Expr *flatten_exprs(google::protobuf::Arena &arena,
                                  const duckdb::vector<vortex::expr::Expr *> &child_filters);
} // namespace vortex
