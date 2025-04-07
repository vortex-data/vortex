#include "expr/expr.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/common/exception.hpp"

#include <cstdint>
#include <duckdb/parser/expression/columnref_expression.hpp>
#include <duckdb/parser/expression/comparison_expression.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/optional_filter.hpp>

using duckdb::ConjunctionAndFilter;
using duckdb::ConstantFilter;
using duckdb::Exception;
using duckdb::ExceptionType;
using duckdb::ExpressionType;
using duckdb::LogicalType;
using duckdb::LogicalTypeId;
using duckdb::TableFilter;
using duckdb::TableFilterType;
using duckdb::Value;
using google::protobuf::Arena;
using std::string;

// vortex expr proto ids.
const string BETWEEN_ID = "between";
const string BINARY_ID = "binary";
const string GET_ITEM_ID = "get_item";
const string IDENTITY_ID = "identity";
const string LIKE_ID = "like";
const string LITERAL_ID = "literal";
const string NOT_ID = "not";

// Temporal ids
const string VORTEX_DATE_ID = "vortex.date";
const string VORTEX_TIME_ID = "vortex.time";
const string VORTEX_TIMESTAMP_ID = "vortex.timestamp";

const string DUCKDB_FUNCTION_NAME_CONTAINS = "contains";

enum TimeUnit : uint8_t {
	/// Nanoseconds
	Ns = 0,
	/// Microseconds
	Us = 1,
	/// Milliseconds
	Ms = 2,
	/// Seconds
	S = 3,
	/// Days
	D = 4,
};

vortex::expr::Kind_BinaryOp into_binary_operation(ExpressionType type) {
	static const std::unordered_map<ExpressionType, vortex::expr::Kind_BinaryOp> op_map = {
	    {ExpressionType::COMPARE_EQUAL, vortex::expr::Kind_BinaryOp_Eq},
	    {ExpressionType::COMPARE_NOTEQUAL, vortex::expr::Kind_BinaryOp_NotEq},
	    {ExpressionType::COMPARE_LESSTHAN, vortex::expr::Kind_BinaryOp_Lt},
	    {ExpressionType::COMPARE_GREATERTHAN, vortex::expr::Kind_BinaryOp_Gt},
	    {ExpressionType::COMPARE_LESSTHANOREQUALTO, vortex::expr::Kind_BinaryOp_Lte},
	    {ExpressionType::COMPARE_GREATERTHANOREQUALTO, vortex::expr::Kind_BinaryOp_Gte},
	    {ExpressionType::CONJUNCTION_AND, vortex::expr::Kind_BinaryOp_And},
	    {ExpressionType::CONJUNCTION_OR, vortex::expr::Kind_BinaryOp_Or}};

	auto value = op_map.find(type);
	if (value == op_map.end()) {
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "into_binary_operation",
		                {{"id", std::to_string(static_cast<uint8_t>(type))}});
	}
	return value->second;
}

TimeUnit timestamp_to_time_unit(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return TimeUnit::S;
	case LogicalTypeId::TIMESTAMP_MS:
		return TimeUnit::Ms;
	case LogicalTypeId::TIMESTAMP:
		return TimeUnit::Us;
	case LogicalTypeId::TIMESTAMP_NS:
		return TimeUnit::Ns;
	default:
		throw Exception(ExceptionType::INVALID, "timestamp_to_time_unit given none timestamp type",
		                {{"id", type.ToString()}});
	}
}

vortex::dtype::DType *into_vortex_dtype(Arena &arena, const LogicalType &type_, bool nullable) {
	auto *dtype = Arena::Create<vortex::dtype::DType>(&arena);
	switch (type_.id()) {
	case LogicalTypeId::INVALID:
	case LogicalTypeId::SQLNULL:
		dtype->mutable_null();
		return dtype;
	case LogicalTypeId::BOOLEAN:
		dtype->mutable_bool_()->set_nullable(nullable);
		return dtype;
	case LogicalTypeId::TINYINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::I8);
		return dtype;
	case LogicalTypeId::SMALLINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::I16);
		return dtype;
	case LogicalTypeId::INTEGER:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::I32);
		return dtype;
	case LogicalTypeId::BIGINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::I64);
		return dtype;
	case LogicalTypeId::UTINYINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::U8);
		return dtype;
	case LogicalTypeId::USMALLINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::U16);
		return dtype;
	case LogicalTypeId::UINTEGER:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::U32);
		return dtype;
	case LogicalTypeId::UBIGINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::U64);
		return dtype;
	case LogicalTypeId::FLOAT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::F32);
		return dtype;
	case LogicalTypeId::DOUBLE:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(vortex::dtype::F64);
		return dtype;
	case LogicalTypeId::CHAR:
	case LogicalTypeId::VARCHAR:
		dtype->mutable_utf8()->set_nullable(nullable);
		return dtype;
	case LogicalTypeId::BLOB:
		dtype->mutable_binary()->set_nullable(nullable);
		return dtype;
	case LogicalTypeId::DATE: {
		dtype->mutable_extension()->set_id(VORTEX_DATE_ID);
		auto storage = dtype->mutable_extension()->mutable_storage_dtype();
		storage->mutable_primitive()->set_nullable(nullable);
		storage->mutable_primitive()->set_type(vortex::dtype::I32);
		dtype->mutable_extension()->set_metadata(std::string({static_cast<uint8_t>(TimeUnit::D)}));
		return dtype;
	}
	case LogicalTypeId::TIME: {
		dtype->mutable_extension()->set_id(VORTEX_TIME_ID);
		auto storage = dtype->mutable_extension()->mutable_storage_dtype();
		storage->mutable_primitive()->set_nullable(nullable);
		storage->mutable_primitive()->set_type(vortex::dtype::I32);
		dtype->mutable_extension()->set_metadata(std::string({static_cast<uint8_t>(TimeUnit::Us)}));
		return dtype;
	}
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_NS: {
		dtype->mutable_extension()->set_id(VORTEX_TIMESTAMP_ID);
		auto storage = dtype->mutable_extension()->mutable_storage_dtype();
		storage->mutable_primitive()->set_nullable(nullable);
		storage->mutable_primitive()->set_type(vortex::dtype::I64);
		auto time_unit = static_cast<char>(timestamp_to_time_unit(type_));
		// This signifies a timestamp without a timezone
		// TODO(joe): support timezones
		dtype->mutable_extension()->set_metadata(std::string({time_unit, 0, 0}));
		return dtype;
	}
	default:
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "into_vortex_dtype", {{"id", type_.ToString()}});
	}
}

vortex::scalar::Scalar *into_null_scalar(Arena &arena, LogicalType &logical_type) {
	auto scalar = Arena::Create<vortex::scalar::Scalar>(&arena);
	scalar->set_allocated_dtype(into_vortex_dtype(arena, logical_type, true));
	scalar->mutable_value()->set_null_value(google::protobuf::NULL_VALUE);
	return scalar;
}

vortex::scalar::Scalar *into_vortex_scalar(Arena &arena, const Value &value, bool nullable) {
	auto scalar = Arena::Create<vortex::scalar::Scalar>(&arena);
	auto dtype = into_vortex_dtype(arena, value.type().id(), nullable);
	scalar->set_allocated_dtype(dtype);

	switch (value.type().id()) {
	case LogicalTypeId::INVALID:
	case LogicalTypeId::SQLNULL: {
		scalar->mutable_value()->set_null_value(google::protobuf::NULL_VALUE);
		return scalar;
	}
	case LogicalTypeId::BOOLEAN: {
		scalar->mutable_value()->set_bool_value(value.GetValue<bool>());
		return scalar;
	}
	case LogicalTypeId::TINYINT:
		scalar->mutable_value()->set_int8_value(value.GetValue<int8_t>());
		return scalar;
	case LogicalTypeId::SMALLINT:
		scalar->mutable_value()->set_int16_value(value.GetValue<int16_t>());
		return scalar;
	case LogicalTypeId::INTEGER:
		scalar->mutable_value()->set_int32_value(value.GetValue<int32_t>());
		return scalar;
	case LogicalTypeId::BIGINT:
		scalar->mutable_value()->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::UTINYINT:
		scalar->mutable_value()->set_uint8_value(value.GetValue<uint8_t>());
		return scalar;
	case LogicalTypeId::USMALLINT:
		scalar->mutable_value()->set_uint16_value(value.GetValue<uint16_t>());
		return scalar;
	case LogicalTypeId::UINTEGER:
		scalar->mutable_value()->set_uint32_value(value.GetValue<uint32_t>());
		return scalar;
	case LogicalTypeId::UBIGINT:
		scalar->mutable_value()->set_uint64_value(value.GetValue<uint64_t>());
		return scalar;
	case LogicalTypeId::FLOAT:
		scalar->mutable_value()->set_f32_value(value.GetValue<float_t>());
		return scalar;
	case LogicalTypeId::DOUBLE:
		scalar->mutable_value()->set_f64_value(value.GetValue<double_t>());
		return scalar;
	case LogicalTypeId::VARCHAR:
		scalar->mutable_value()->set_string_value(value.GetValue<string>());
		return scalar;
	case LogicalTypeId::DATE:
		scalar->mutable_value()->set_int32_value(value.GetValue<int32_t>());
		return scalar;
	case LogicalTypeId::TIME:
		scalar->mutable_value()->set_int32_value(value.GetValue<int32_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP_SEC:
		scalar->mutable_value()->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP_MS:
		scalar->mutable_value()->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP:
		scalar->mutable_value()->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP_NS:
		scalar->mutable_value()->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	default:
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "into_vortex_scalar", {{"id", value.ToString()}});
	}
}

void set_column(const string &s, vortex::expr::Expr *column) {
	column->set_id(GET_ITEM_ID);
	auto kind = column->mutable_kind();
	auto get_item = kind->mutable_get_item();
	get_item->mutable_path()->assign(s);

	auto id = column->add_children();
	id->mutable_kind()->mutable_identity();
	id->set_id(IDENTITY_ID);
}

void set_literal(Arena &arena, const Value &value, bool nullable, vortex::expr::Expr *constant) {
	auto literal = constant->mutable_kind()->mutable_literal();
	auto dvalue = into_vortex_scalar(arena, value, nullable);
	literal->set_allocated_value(dvalue);
	constant->set_id(LITERAL_ID);
}

vortex::expr::Expr *flatten_table_filters(Arena &arena, duckdb::vector<duckdb::unique_ptr<TableFilter>> &child_filters,
                                          const string &column_name) {
	D_ASSERT(!child_filters.empty());

	if (child_filters.size() == 1) {
		return table_expression_into_expr(arena, *child_filters[0], column_name);
	}

	// Start with the first expression
	auto tail = static_cast<vortex::expr::Expr *>(nullptr);
	auto hd = Arena::Create<vortex::expr::Expr>(&arena);

	// Flatten the list of children into a linked list of AND values.
	for (size_t i = 0; i < child_filters.size() - 1; i++) {
		vortex::expr::Expr *new_and = !tail ? hd : tail->add_children();
		new_and->set_id(BINARY_ID);
		new_and->mutable_kind()->set_binary_op(vortex::expr::Kind::And);
		new_and->add_children()->Swap(table_expression_into_expr(arena, *child_filters[i], column_name));

		tail = new_and;
	}
	tail->add_children()->Swap(table_expression_into_expr(arena, *child_filters.back(), column_name));
	return hd;
}

vortex::expr::Expr *flatten_exprs(Arena &arena, const duckdb::vector<vortex::expr::Expr *> &child_filters) {

	if (child_filters.empty()) {
		auto expr = arena.Create<vortex::expr::Expr>(&arena);
		set_literal(arena, Value(true), true, expr);
		return expr;
	}

	if (child_filters.size() == 1) {
		return child_filters[0];
	}

	// Start with the first expression
	auto tail = static_cast<vortex::expr::Expr *>(nullptr);
	auto hd = Arena::Create<vortex::expr::Expr>(&arena);

	// Flatten the list of children into a linked list of AND values.
	for (size_t i = 0; i < child_filters.size() - 1; i++) {
		vortex::expr::Expr *new_and = !tail ? hd : tail->add_children();
		new_and->set_id(BINARY_ID);
		new_and->mutable_kind()->set_binary_op(vortex::expr::Kind::And);
		new_and->add_children()->Swap(child_filters[i]);

		tail = new_and;
	}
	tail->add_children()->Swap(child_filters.back());
	return hd;
}

std::optional<string> expr_to_like_pattern(const duckdb::Expression &dexpr) {
	switch (dexpr.expression_class) {
	case duckdb::ExpressionClass::BOUND_CONSTANT: {
		auto &dconstant = dexpr.Cast<duckdb::BoundConstantExpression>();
		auto contains_pattern = dconstant.value.GetValue<string>();
		auto like_pattern = "%" + contains_pattern + "%";
		return std::optional(like_pattern);
	};
	default:
		return std::nullopt;
	}
}

vortex::expr::Expr *expression_into_vortex_expr(Arena &arena, const duckdb::Expression &dexpr) {
	auto expr = Arena::Create<vortex::expr::Expr>(&arena);
	switch (dexpr.expression_class) {
	case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
		auto &dcol_ref = dexpr.Cast<duckdb::BoundColumnRefExpression>();
		auto column = expr;
		set_column(dcol_ref.GetName(), column);
		return expr;
	}
	case duckdb::ExpressionClass::BOUND_CONSTANT: {
		auto &dconstant = dexpr.Cast<duckdb::BoundConstantExpression>();
		set_literal(arena, Value(dconstant.value), true, expr);
		return expr;
	}
	case duckdb::ExpressionClass::BOUND_COMPARISON: {
		auto &dcompare = dexpr.Cast<duckdb::BoundComparisonExpression>();
		auto left = expr->add_children();
		left->Swap(expression_into_vortex_expr(arena, *dcompare.left));
		auto right = expr->add_children();
		right->Swap(expression_into_vortex_expr(arena, *dcompare.right));
		auto bin_op = into_binary_operation(dcompare.type);
		expr->mutable_kind()->set_binary_op(bin_op);
		expr->set_id(BINARY_ID);
		return expr;
	}
	case duckdb::ExpressionClass::BOUND_BETWEEN: {
		auto &dbetween = dexpr.Cast<duckdb::BoundBetweenExpression>();
		auto col = expression_into_vortex_expr(arena, *dbetween.input);
		auto lower = expression_into_vortex_expr(arena, *dbetween.lower);
		auto upper = expression_into_vortex_expr(arena, *dbetween.upper);
		// Between order on vx is arr, lower, upper.
		expr->add_children()->Swap(col);
		expr->add_children()->Swap(lower);
		expr->add_children()->Swap(upper);
		auto kind = expr->mutable_kind()->mutable_between();
		kind->set_lower_strict(!dbetween.lower_inclusive);
		kind->set_upper_strict(!dbetween.upper_inclusive);
		expr->set_id(BETWEEN_ID);
		return expr;
	}
	case duckdb::ExpressionClass::BOUND_OPERATOR: {
		auto &dop = dexpr.Cast<duckdb::BoundOperatorExpression>();
		if (dop.type != ExpressionType::OPERATOR_NOT) {
			return nullptr;
		}
		auto child = expr->add_children();
		auto fn = expression_into_vortex_expr(arena, *dop.children[0]);
		if (fn == nullptr) {
			return nullptr;
		}
		child->Swap(fn);
		expr->mutable_kind()->mutable_not_();
		expr->set_id(NOT_ID);
		return expr;
	}
	case duckdb::ExpressionClass::BOUND_FUNCTION: {
		auto &dfunc_expr = dexpr.Cast<duckdb::BoundFunctionExpression>();
		auto &dfunc = dfunc_expr.function;
		if (dfunc.name == DUCKDB_FUNCTION_NAME_CONTAINS) {
			assert(dfunc_expr.children.size() == 2);
			// value
			expr->add_children()->Swap(expression_into_vortex_expr(arena, *dfunc_expr.children[0]));
			// pattern
			auto pattern = expr->add_children();

			auto pattern_value = expr_to_like_pattern(*dfunc_expr.children[1]);
			if (!pattern_value.has_value()) {
				return nullptr;
			}
			set_literal(arena, Value(pattern_value.value()), true, pattern);
			auto like = expr->mutable_kind()->mutable_like();
			like->set_case_insensitive(false);
			like->set_negated(false);
			expr->set_id(LIKE_ID);
			return expr;
		}
		return nullptr;
	}
	default:
		return nullptr;
	}
}

vortex::expr::Expr *table_expression_into_expr(Arena &arena, TableFilter &filter, const string &column_name) {
	auto expr = Arena::Create<vortex::expr::Expr>(&arena);
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto bin_op = into_binary_operation(constant_filter.comparison_type);

		set_column(column_name, expr->add_children());
		set_literal(arena, constant_filter.constant, true, expr->add_children());

		expr->mutable_kind()->set_binary_op(bin_op);
		expr->set_id(BINARY_ID);
		return expr;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjucts = filter.Cast<ConjunctionAndFilter>();

		return flatten_table_filters(arena, conjucts.child_filters, column_name);
	}
	case TableFilterType::IS_NULL:
	case TableFilterType::IS_NOT_NULL: {
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "null checks");
	}
	case TableFilterType::OPTIONAL_FILTER: {
		expr->set_id(LITERAL_ID);
		auto lit = expr->mutable_kind()->mutable_literal();
		lit->mutable_value()->mutable_value()->set_bool_value(true);
		lit->mutable_value()->mutable_dtype()->mutable_bool_()->set_nullable(false);
		return expr;
	}
	default:
		break;
	}
	throw Exception(ExceptionType::NOT_IMPLEMENTED, "table_expression_into_expr",
	                {{"filter_type_id", std::to_string(static_cast<uint8_t>(filter.filter_type))}});
}
