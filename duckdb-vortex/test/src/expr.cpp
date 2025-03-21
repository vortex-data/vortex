#include "catch2/catch_test_macros.hpp"
#include "duckdb.hpp"
#include "expr/expr.hpp"

#include <iostream>
#include <duckdb/planner/filter/constant_filter.hpp>

TEST_CASE("Test DuckDB list handling", "[list]") {

	google::protobuf::Arena arena;

	auto filter = duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value(1));
	auto filter2 = duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value(2));
	auto filter3 = duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value(3));

	auto filter_and = duckdb::ConjunctionAndFilter();

	filter_and.child_filters.push_back(std::move(filter));
	filter_and.child_filters.push_back(std::move(filter2));
	filter_and.child_filters.push_back(std::move(filter3));

	auto col = std::string("a");
	auto expr = table_expression_into_expr(arena, filter_and, col);

	REQUIRE(expr->children().size() == 2);
	REQUIRE(expr->kind().binary_op() == vortex::expr::Kind_BinaryOp_And);

	REQUIRE(expr->children()[0].children().size() == 2);
	REQUIRE(expr->children()[0].kind().binary_op() == vortex::expr::Kind_BinaryOp_Eq);
}
