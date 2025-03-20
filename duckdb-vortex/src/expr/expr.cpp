
#include "expr/expr.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/common/exception.hpp"

#include <duckdb/planner/filter/conjunction_filter.hpp>

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

vortex::expr::Kind_BinaryOp into_binary_operation(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
		return vortex::expr::Kind_BinaryOp_Eq;
	case ExpressionType::COMPARE_NOTEQUAL:
		return vortex::expr::Kind_BinaryOp_NotEq;
	case ExpressionType::COMPARE_LESSTHAN:
		return vortex::expr::Kind_BinaryOp_Lt;
	case ExpressionType::COMPARE_GREATERTHAN:
		return vortex::expr::Kind_BinaryOp_Gt;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return vortex::expr::Kind_BinaryOp_Lte;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return vortex::expr::Kind_BinaryOp_Gte;
	case ExpressionType::CONJUNCTION_AND:
		return vortex::expr::Kind_BinaryOp_And;
	case ExpressionType::CONJUNCTION_OR:
		return vortex::expr::Kind_BinaryOp_Or;
	default:
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "impl");
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
		dtype->mutable_binary()->set_nullable(nullable);
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
	default:
		break;
	}
	throw Exception(ExceptionType::NOT_IMPLEMENTED, "into_vortex_dtype", {{"id", type_.ToString()}});
}

vortex::scalar::Scalar *into_null_scalar(Arena &arena, LogicalType &type_) {
	auto scalar = Arena::Create<vortex::scalar::Scalar>(&arena);
	scalar->set_allocated_dtype(into_vortex_dtype(arena, type_, true));
	scalar->mutable_value()->set_null_value(google::protobuf::NULL_VALUE);
	return scalar;
}

vortex::scalar::Scalar *into_vortex_scalar(Arena &arena, Value &value, bool nullable) {
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
		auto boolean = new vortex::dtype::Bool();
		boolean->set_nullable(nullable);
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
	default:
		break;
	}
	throw Exception(ExceptionType::NOT_IMPLEMENTED, "into_vortex_scalar", {{"id", value.ToString()}});
}

void set_column(const string &s, vortex::expr::Expr *column) {
	auto get_item = new vortex::expr::Kind_GetItem();
	get_item->mutable_path()->assign(s);
	auto kind = column->mutable_kind();
	kind->set_allocated_get_item(get_item);
	column->add_children()->mutable_kind()->set_allocated_identity(new vortex::expr::Kind_Identity());
}

vortex::expr::Expr *table_expression_into_expr(Arena &arena, TableFilter &filter, string &column_name) {
	auto expr = Arena::Create<vortex::expr::Expr>(&arena);
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto bin_op = into_binary_operation(constant_filter.comparison_type);
		auto value = into_vortex_scalar(arena, constant_filter.constant, true);

		auto column = expr->add_children();
		set_column(column_name, column);

		auto constant = expr->add_children()->mutable_kind();
		auto literal = constant->mutable_literal();
		literal->set_allocated_value(value);

		expr->mutable_kind()->set_binary_op(bin_op);
		expr->set_id("binary");
		return expr;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjucts = filter.Cast<ConjunctionAndFilter>();

		D_ASSERT(conjucts.child_filters.size() > 1);

		// Start with the first expression
		auto tail = static_cast<vortex::expr::Expr *>(nullptr);
		auto hd = Arena::Create<vortex::expr::Expr>(&arena);

		// Flatten the list of children into a linked list of AND values.
		for (size_t i = 0; i < conjucts.child_filters.size(); i++) {
			vortex::expr::Expr *new_and;
			if (!tail) {
				new_and = hd;
			} else {
				new_and = tail->add_children();
			}
			new_and->set_id("binary");
			new_and->mutable_kind()->set_binary_op(vortex::expr::Kind::And);
			new_and->add_children()->Swap(table_expression_into_expr(arena, *conjucts.child_filters[i], column_name));

			tail = new_and;
		}
		tail->add_children()->Swap(table_expression_into_expr(arena, *conjucts.child_filters.back(), column_name));
		return hd;
	}
	case TableFilterType::IS_NULL:
	case TableFilterType::IS_NOT_NULL: {
		throw Exception(ExceptionType::NOT_IMPLEMENTED, "null checks");
	}
	default:
		break;
	}

	throw Exception(ExceptionType::NOT_IMPLEMENTED, "table_expression_into_expr",
	                {{"filter_type_id", std::to_string(static_cast<uint8_t>(filter.filter_type))}});
}
