#include <cstdint>

#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"

#include "vortex_expr.hpp"

#include "duckdb/planner/filter/in_filter.hpp"

using duckdb::ConjunctionAndFilter;
using duckdb::ConjunctionOrFilter;
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

namespace vortex {

// Temporal ids
const string VORTEX_DATE_ID = "vortex.date";
const string VORTEX_TIME_ID = "vortex.time";
const string VORTEX_TIMESTAMP_ID = "vortex.timestamp";

const string VX_ROW_ID_COL_ID = "$vx.row_id";

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

expr::Kind_BinaryOp into_binary_operation(ExpressionType type) {
	static const std::unordered_map<ExpressionType, expr::Kind_BinaryOp> op_map = {
	    {ExpressionType::COMPARE_EQUAL, expr::Kind_BinaryOp_Eq},
	    {ExpressionType::COMPARE_NOTEQUAL, expr::Kind_BinaryOp_NotEq},
	    {ExpressionType::COMPARE_LESSTHAN, expr::Kind_BinaryOp_Lt},
	    {ExpressionType::COMPARE_GREATERTHAN, expr::Kind_BinaryOp_Gt},
	    {ExpressionType::COMPARE_LESSTHANOREQUALTO, expr::Kind_BinaryOp_Lte},
	    {ExpressionType::COMPARE_GREATERTHANOREQUALTO, expr::Kind_BinaryOp_Gte},
	    {ExpressionType::CONJUNCTION_AND, expr::Kind_BinaryOp_And},
	    {ExpressionType::CONJUNCTION_OR, expr::Kind_BinaryOp_Or}};

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

dtype::DType *into_vortex_dtype(Arena &arena, const LogicalType &type_, bool nullable) {
	auto *dtype = Arena::Create<dtype::DType>(&arena);
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
		dtype->mutable_primitive()->set_type(dtype::I8);
		return dtype;
	case LogicalTypeId::SMALLINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::I16);
		return dtype;
	case LogicalTypeId::INTEGER:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::I32);
		return dtype;
	case LogicalTypeId::BIGINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::I64);
		return dtype;
	case LogicalTypeId::UTINYINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::U8);
		return dtype;
	case LogicalTypeId::USMALLINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::U16);
		return dtype;
	case LogicalTypeId::UINTEGER:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::U32);
		return dtype;
	case LogicalTypeId::UBIGINT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::U64);
		return dtype;
	case LogicalTypeId::FLOAT:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::F32);
		return dtype;
	case LogicalTypeId::DOUBLE:
		dtype->mutable_primitive()->set_nullable(nullable);
		dtype->mutable_primitive()->set_type(dtype::F64);
		return dtype;
	case LogicalTypeId::DECIMAL: {
		dtype->mutable_decimal()->set_nullable(nullable);
		auto decimal = dtype->mutable_decimal();
		decimal->set_precision(duckdb::DecimalType::GetWidth(type_));
		decimal->set_scale(duckdb::DecimalType::GetScale(type_));
		return dtype;
	}
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
		storage->mutable_primitive()->set_type(dtype::I32);
		dtype->mutable_extension()->set_metadata(std::string({static_cast<uint8_t>(TimeUnit::D)}));
		return dtype;
	}
	case LogicalTypeId::TIME: {
		dtype->mutable_extension()->set_id(VORTEX_TIME_ID);
		auto storage = dtype->mutable_extension()->mutable_storage_dtype();
		storage->mutable_primitive()->set_nullable(nullable);
		storage->mutable_primitive()->set_type(dtype::I32);
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
		storage->mutable_primitive()->set_type(dtype::I64);
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

scalar::Scalar *into_null_scalar(Arena &arena, LogicalType &logical_type) {
	auto scalar = Arena::Create<scalar::Scalar>(&arena);
	scalar->set_allocated_dtype(into_vortex_dtype(arena, logical_type, true));
	scalar->mutable_value()->set_null_value(google::protobuf::NULL_VALUE);
	return scalar;
}

scalar::ScalarValue *into_vortex_scalar_value(Arena &arena, const Value &value) {
	auto scalar = Arena::Create<scalar::ScalarValue>(&arena);

	switch (value.type().id()) {
	case LogicalTypeId::INVALID:
	case LogicalTypeId::SQLNULL: {
		scalar->set_null_value(google::protobuf::NULL_VALUE);
		return scalar;
	}
	case LogicalTypeId::BOOLEAN: {
		scalar->set_bool_value(value.GetValue<bool>());
		return scalar;
	}
	case LogicalTypeId::TINYINT:
		scalar->set_int64_value(value.GetValue<int8_t>());
		return scalar;
	case LogicalTypeId::SMALLINT:
		scalar->set_int64_value(value.GetValue<int16_t>());
		return scalar;
	case LogicalTypeId::INTEGER:
		scalar->set_int64_value(value.GetValue<int32_t>());
		return scalar;
	case LogicalTypeId::BIGINT:
		scalar->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::UTINYINT:
		scalar->set_uint64_value(value.GetValue<uint8_t>());
		return scalar;
	case LogicalTypeId::USMALLINT:
		scalar->set_uint64_value(value.GetValue<uint16_t>());
		return scalar;
	case LogicalTypeId::UINTEGER:
		scalar->set_uint64_value(value.GetValue<uint32_t>());
		return scalar;
	case LogicalTypeId::UBIGINT:
		scalar->set_uint64_value(value.GetValue<uint64_t>());
		return scalar;
	case LogicalTypeId::FLOAT:
		scalar->set_f32_value(value.GetValue<float_t>());
		return scalar;
	case LogicalTypeId::DOUBLE:
		scalar->set_f64_value(value.GetValue<double_t>());
		return scalar;
	case LogicalTypeId::DECIMAL: {
		auto huge = value.GetValue<duckdb::hugeint_t>();
		uint32_t out[4];
		out[0] = static_cast<uint32_t>(huge);
		out[1] = static_cast<uint32_t>(huge >> 32);
		out[2] = static_cast<uint32_t>(huge >> 64);
		out[3] = static_cast<uint32_t>(huge >> 96);
		scalar->set_bytes_value(std::string(reinterpret_cast<char *>(out), 8));
		return scalar;
	}
	case LogicalTypeId::VARCHAR:
		scalar->set_string_value(value.GetValue<string>());
		return scalar;
	case LogicalTypeId::DATE:
		scalar->set_int64_value(value.GetValue<int32_t>());
		return scalar;
	case LogicalTypeId::TIME:
		scalar->set_int64_value(value.GetValue<int32_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP_SEC:
		scalar->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP_MS:
		scalar->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP:
		scalar->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	case LogicalTypeId::TIMESTAMP_NS:
		scalar->set_int64_value(value.GetValue<int64_t>());
		return scalar;
	default:
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "into_vortex_scalar", {{"id", value.ToString()}});
	}
}

scalar::Scalar *into_vortex_scalar(Arena &arena, const Value &value, bool nullable) {
	auto scalar = Arena::Create<scalar::Scalar>(&arena);
	auto dtype = into_vortex_dtype(arena, value.type(), nullable);
	scalar->set_allocated_dtype(dtype);

	auto scalar_value = into_vortex_scalar_value(arena, value);
	scalar->mutable_value()->Swap(scalar_value);
	return scalar;
}

void set_column(const string &s, expr::Expr *column) {

	column->set_id(GET_ITEM_ID);
	auto kind = column->mutable_kind();
	auto get_item = kind->mutable_get_item();
	get_item->mutable_path()->assign(s);

	auto id = column->add_children();
	id->set_id(VAR_ID);
	if (s == "file_row_number" || s == "file_index") {
		id->mutable_kind()->mutable_var()->set_var(VX_ROW_ID_COL_ID);
	} else {
		id->mutable_kind()->mutable_var()->set_var("");
	}
}

void set_literal(Arena &arena, const Value &value, bool nullable, expr::Expr *constant) {
	auto literal = constant->mutable_kind()->mutable_literal();
	auto dvalue = into_vortex_scalar(arena, value, nullable);
	literal->set_allocated_value(dvalue);
	constant->set_id(LITERAL_ID);
}

expr::Expr *flatten_table_filters(Arena &arena, duckdb::vector<duckdb::unique_ptr<TableFilter>> &child_filters,
                                  expr::Kind_BinaryOp operation, const string &column_name) {

	D_ASSERT(!child_filters.empty());

	if (child_filters.size() == 1) {
		return table_expression_into_expr(arena, *child_filters[0], column_name);
	}

	// Start with the first expression
	auto tail = static_cast<expr::Expr *>(nullptr);
	auto hd = Arena::Create<expr::Expr>(&arena);

	// Flatten the list of children into a linked list of operation values.
	for (size_t i = 0; i < child_filters.size() - 1; i++) {
		expr::Expr *new_and = !tail ? hd : tail->add_children();
		new_and->set_id(BINARY_ID);
		new_and->mutable_kind()->set_binary_op(operation);
		new_and->add_children()->Swap(table_expression_into_expr(arena, *child_filters[i], column_name));

		tail = new_and;
	}
	tail->add_children()->Swap(table_expression_into_expr(arena, *child_filters.back(), column_name));
	return hd;
}

expr::Expr *flatten_exprs(Arena &arena, const duckdb::vector<vortex::expr::Expr *> &child_filters) {

	if (child_filters.empty()) {
		auto expr = arena.Create<expr::Expr>(&arena);
		set_literal(arena, Value(true), true, expr);
		return expr;
	}

	if (child_filters.size() == 1) {
		return child_filters[0];
	}

	// Start with the first expression
	auto tail = static_cast<expr::Expr *>(nullptr);
	auto hd = Arena::Create<expr::Expr>(&arena);

	// Flatten the list of children into a linked list of AND values.
	for (size_t i = 0; i < child_filters.size() - 1; i++) {
		expr::Expr *new_and = !tail ? hd : tail->add_children();
		new_and->set_id(BINARY_ID);
		new_and->mutable_kind()->set_binary_op(expr::Kind::And);
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

expr::Expr *expression_into_vortex_expr(Arena &arena, const duckdb::Expression &dexpr) {
	auto expr = Arena::Create<expr::Expr>(&arena);
	switch (dexpr.expression_class) {
	case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
		auto &dcol_ref = dexpr.Cast<duckdb::BoundColumnRefExpression>();
		auto column = expr;
		set_column(dcol_ref.GetName(), column);
		return expr;
	}
	case duckdb::ExpressionClass::BOUND_CONSTANT: {
		auto &dconstant = dexpr.Cast<duckdb::BoundConstantExpression>();
		set_literal(arena, dconstant.value, true, expr);
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

void set_list_element(Arena &arena, expr::Expr *list, duckdb::vector<Value> &values) {
	list->set_id(LITERAL_ID);
	auto ll = list->mutable_kind()->mutable_literal();
	auto scalar = ll->mutable_value();
	auto elem_type = into_vortex_dtype(arena, values[0].GetTypeMutable(), true);
	auto list_type = scalar->mutable_dtype()->mutable_list();
	list_type->mutable_element_type()->Swap(elem_type);
	list_type->set_nullable(true);
	auto list_scalar_value = scalar->mutable_value()->mutable_list_value();
	for (auto &elem : values) {
		list_scalar_value->mutable_values()->Add(std::move(*into_vortex_scalar_value(arena, elem)));
	}
}

expr::Expr *table_expression_into_expr(Arena &arena, TableFilter &filter, const string &column_name) {
	auto expr = Arena::Create<expr::Expr>(&arena);
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
	case TableFilterType::CONJUNCTION_OR: {
		auto &disjuncts = filter.Cast<ConjunctionOrFilter>();
		return flatten_table_filters(arena, disjuncts.child_filters, expr::Kind_BinaryOp_Or, column_name);
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjucts = filter.Cast<ConjunctionAndFilter>();

		return flatten_table_filters(arena, conjucts.child_filters, expr::Kind_BinaryOp_And, column_name);
	}
	case TableFilterType::IS_NULL:
	case TableFilterType::IS_NOT_NULL: {
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "null checks");
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto *expr_o =
		    table_expression_into_expr(arena, *filter.Cast<duckdb::OptionalFilter>().child_filter, column_name);
		if (expr_o != nullptr) {
			return expr_o;
		}
		expr->set_id(LITERAL_ID);
		auto lit = expr->mutable_kind()->mutable_literal();
		lit->mutable_value()->mutable_value()->set_bool_value(true);
		lit->mutable_value()->mutable_dtype()->mutable_bool_()->set_nullable(false);
		return expr;
	}
	case TableFilterType::IN_FILTER: {
		auto &in_list_filter = filter.Cast<duckdb::InFilter>();
		expr->set_id(LIST_CONTAINS_ID);
		expr->mutable_kind()->mutable_list_contains();
		auto list = expr->add_children();
		set_list_element(arena, list, in_list_filter.values);
		set_column(column_name, expr->add_children());
		return expr;
	}
	default:
		break;
	}
	std::cout << "table expr: " << std::to_string(static_cast<uint8_t>(filter.filter_type)) << filter.DebugToString()
	          << std::endl;
	throw Exception(ExceptionType::NOT_IMPLEMENTED, "table_expression_into_expr",
	                {{"filter_type_id", std::to_string(static_cast<uint8_t>(filter.filter_type))}});
}

vortex::expr::Expr *pack_projection_columns(google::protobuf::Arena &arena, duckdb::vector<std::string> columns) {
	auto expr = arena.Create<expr::Expr>(&arena);
	expr->set_id(PACK_ID);
	auto pack_paths = expr->mutable_kind()->mutable_pack()->mutable_paths();
	for (auto &columnn : columns) {
		set_column(columnn, expr->add_children());
		pack_paths->Add(std::string(columnn));
	}

	return expr;
}

} // namespace vortex
