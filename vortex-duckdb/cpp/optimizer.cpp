#include "duckdb_vx/optimizer.h"
#include "duckdb_vx/duckdb_diagnostics.h"
DUCKDB_INCLUDES_BEGIN
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
DUCKDB_INCLUDES_END

using namespace duckdb;

/*
 * Until https://github.com/duckdb/duckdb/pull/22788 is merged, and Duckdb
 * version used in Vortex is bumped to include this, we'll have our separate
 * optimizer pass pushing down types to Vortex.
 */

// Collect CAST(bound_column, T) patterns where bound_column binds into given GET's index.
void CollectCastTypes(const Expression &expr, idx_t index, const vector<ColumnIndex> &column_ids,
                      unordered_map<column_t, LogicalType> &cast_map, unordered_set<column_t> &conflicts) {
	auto collect_children = [&] {
		ExpressionIterator::EnumerateChildren(
		    expr, [&](const Expression &child) { CollectCastTypes(child, index, column_ids, cast_map, conflicts); });
	};

	if (expr.GetExpressionClass() != ExpressionClass::BOUND_CAST) {
		return collect_children();
	}
	auto &bound_cast = expr.Cast<BoundCastExpression>();

	if (bound_cast.child->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		return collect_children();
	}
	auto &bound_column = bound_cast.child->Cast<BoundColumnRefExpression>();

	if (bound_column.depth > 0 || bound_column.binding.table_index != index) {
		return collect_children();
	}

	// We are in a leaf
	const column_t projection_id = bound_column.binding.column_index;
	if (IsVirtualColumn(projection_id)) {
		return;
	}
	D_ASSERT(projection_id < column_ids.size());
	const column_t column_id = column_ids[projection_id].GetPrimaryIndex();
	if (auto it = cast_map.find(column_id); it == cast_map.end()) {
		cast_map.emplace(column_id, bound_cast.return_type);
	} else if (it->second != bound_cast.return_type) {
		conflicts.insert(column_id);
	}
}

// Replace every CAST(bound_column, T) with a bare bound_column at type T when T
// is listed in projection_cast.
static void ReplaceCastTypes(unique_ptr<Expression> &expr, idx_t index,
                             const unordered_map<column_t, LogicalType> &projection_cast) {
	auto replace_children = [&] {
		ExpressionIterator::EnumerateChildren(
		    *expr, [&](unique_ptr<Expression> &child) { ReplaceCastTypes(child, index, projection_cast); });
	};

	if (expr->GetExpressionClass() != ExpressionClass::BOUND_CAST) {
		return replace_children();
	}
	auto &bound_cast = expr->Cast<BoundCastExpression>();

	if (bound_cast.child->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		return replace_children();
	}
	auto &bound_column = bound_cast.child->Cast<BoundColumnRefExpression>();

	if (bound_column.depth > 0 || bound_column.binding.table_index != index) {
		return replace_children();
	}

	const column_t projection_id = bound_column.binding.column_index;
	auto it = projection_cast.find(projection_id);
	if (it == projection_cast.end() || it->second != bound_cast.return_type) {
		return replace_children();
	}

	expr = make_uniq<BoundColumnRefExpression>(it->second, bound_column.binding);
}

// Walk the plan bottom-up and, for each node whose direct child is a GET that
// supports type_pushdown, push every CAST(colref, T) found in that node's
// expressions into the GET so the scan produces T directly.
unique_ptr<LogicalOperator> TryPushdownCastTypes(ClientContext& context, unique_ptr<LogicalOperator> op) {
	for (auto &child : op->children) {
		child = TryPushdownCastTypes(context, std::move(child));
	}

	for (const auto &child : op->children) {
		if (child->type != LogicalOperatorType::LOGICAL_GET) {
			continue;
		}
		auto &get = child->Cast<LogicalGet>();
		if (!get.function.type_pushdown) {
			continue;
		}

		const vector<ColumnIndex> &column_ids = get.GetColumnIds();
		const idx_t index = get.table_index;
		unordered_map<column_t, LogicalType> cast_map;
		unordered_set<column_t> conflicts;

		LogicalOperatorVisitor::EnumerateExpressions(*op, [&](unique_ptr<Expression> *expr_ptr) {
			CollectCastTypes(**expr_ptr, index, column_ids, cast_map, conflicts);
		});

		for (column_t col_id : conflicts) {
			cast_map.erase(col_id);
		}
		if (cast_map.empty()) {
			continue;
		}

		get.function.type_pushdown(context, get.bind_data, cast_map);
		for (const auto &[col_id, new_type] : cast_map) {
			get.returned_types[col_id] = new_type;
		}

		unordered_map<idx_t, LogicalType> proj_to_type;
		for (idx_t i = 0; i < column_ids.size(); i++) {
			const column_t col_idx = column_ids[i].GetPrimaryIndex();
			if (auto it = cast_map.find(col_idx); it != cast_map.end()) {
				proj_to_type[i] = it->second;
			}
		}

		LogicalOperatorVisitor::EnumerateExpressions(
		    *op, [&](unique_ptr<Expression> *expr_ptr) { ReplaceCastTypes(*expr_ptr, get.table_index, proj_to_type); });
	}

	return op;
}

static void VortexOptimizeFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
    plan = TryPushdownCastTypes(input.context, std::move(plan));
}

class VortexOptimizerExtension final : public OptimizerExtension {
public:
	VortexOptimizerExtension() {
		optimize_function = VortexOptimizeFunction;
	}
};

extern "C" duckdb_state duckdb_vx_optimizer_extension_register(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;
    try {
        DBConfig::GetConfig(db).GetCallbackManager().Register(VortexOptimizerExtension());
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex optimizer extension:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}
