// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb_vx/optimizer.h"
#include "duckdb_vx/table_function.h"
#include "vortex.h"
#include <optional>

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

void VortexOptimizeFunction(OptimizerExtensionInput &input, LogicalOperatorPtr &plan) {
    plan = TryPushdownScalarFunctions(input.context, std::move(plan));
}

LogicalOperatorPtr TryPushdownScalarFunctions(ClientContext &context, LogicalOperatorPtr plan) {
    Analyses analyses;
    Projections projections;
    FindGetsAndAliases(*plan, analyses, projections);
    if (analyses.empty()) {
        return plan;
    }

    ScalarFnCollect(analyses, projections).VisitOperator(*plan);

    bool any_pushed = false;
    for (auto &[_, analysis] : analyses) {
        for (auto &[column_index, expr] : analysis.col_to_fn) {
            if (expr == nullptr) { // Conflict for column
                continue;
            }
            const TableColumnStorageIndex storage_index = analysis.storageIndex(column_index);
            TableFunctionProjectionExpressionInput input {analysis.get, *expr, storage_index};
            if (projection_expression_pushdown(context, input)) {
                analysis.get.types[column_index] = expr->return_type;
                analysis.get.returned_types[storage_index] = expr->return_type;
                any_pushed = true;
            } else { // failed to push down expression, can't replace it
                expr = nullptr;
            }
        }
    }

    if (any_pushed) {
        ScalarFnReplace(analyses, projections).VisitOperator(*plan);
    }
    return plan;
}

void FindGetsAndAliases(LogicalOperator &op,
                        Analyses &analyses,
                        Projections &projections,
                        LogicalOperator *parent) {
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &get = op.Cast<LogicalGet>();
        if (get.function.bind == bind) {
            std::cout << "Registered Vortex binder with table index " << get.table_index << "\n";

            analyses.emplace(get.table_index, GetAnalysis {get, {}});
            if (parent && parent->type == LogicalOperatorType::LOGICAL_PROJECTION) {
                const auto &projection = parent->Cast<LogicalProjection>();
                projections.emplace(projection.table_index, projection);

                std::cout << "Registered Vortex PROJECTION with table index " << projection.table_index
                          << "\n";
            }
        }
    }
    for (auto &child : op.children) {
        FindGetsAndAliases(*child, analyses, projections, &op);
    }
}

struct Binding {
    GetAnalysis &analysis;
    TableColumnScanIndex column_index;
};

/*
 * For a given (table index, column index) pair, resolve it to a GET and a
 * GET's column scan index.
 * Returns nullopt for virtual columns and columns which are neither part of
 * GET nor part of PROJECTION wrapping a GET.
 */
static std::optional<Binding>
Resolve(ColumnBinding binding, Analyses &analyses, const Projections &projections) {
    if (IsVirtualColumn(binding.column_index)) {
        return std::nullopt;
    }
    if (const auto it = analyses.find(binding.table_index); it != analyses.end()) {
        const TableColumnStorageIndex storage_index = it->second.storageIndex(binding.column_index);
        std::cout << StringUtil::Format(
            "Binding %s is SELECT from Vortex with storage index %d, name %s, returned type %s, type %s\n",
            binding.ToString(),
            storage_index,
            it->second.get.names[storage_index],
            EnumUtil::ToString(it->second.get.returned_types[storage_index].id()),
            EnumUtil::ToString(it->second.get.types[binding.column_index].id()));
        return {{it->second, binding.column_index}};
    }

    const auto projection_it = projections.find(binding.table_index);
    if (projection_it == projections.end()) {
        return std::nullopt;
    }

    const ExpressionPtr &inner = projection_it->second.expressions[binding.column_index];
    if (inner->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::nullopt;
    }
    const ColumnBinding &get_binding = inner->Cast<BoundColumnRefExpression>().binding;
    if (IsVirtualColumn(get_binding.column_index)) {
        return std::nullopt;
    }
    if (const auto it = analyses.find(get_binding.table_index); it != analyses.end()) {
        const TableColumnStorageIndex storage_index = it->second.storageIndex(get_binding.column_index);
        std::cout << StringUtil::Format(
            "Binding %s is SELECT from PROJECTION with GET binding %s, storage index %d, name "
            "%s, returned type %s, type %s\n",
            binding.ToString(),
            get_binding.ToString(),
            storage_index,
            it->second.get.names[storage_index],
            EnumUtil::ToString(it->second.get.returned_types[storage_index].id()),
            EnumUtil::ToString(it->second.get.types[get_binding.column_index].id()));
        return {{it->second, get_binding.column_index}};
    }
    return std::nullopt;
}

void ScalarFnCollect::VisitOperator(LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_PROJECTION &&
        projections.count(op.Cast<LogicalProjection>().table_index)) {
        // Logical projection expressions are columns which reference underlying
        // GETs. Don't process them, as they would add conflicts for every
        // column used in projection. Example:
        // PROJECTION(col) -> GET(col). We don't want to visit
        // BoundColumnRefExpression in PROJECTION.
        VisitOperatorChildren(op);
        return;
    }
    LogicalOperatorVisitor::VisitOperator(op);
}

ExpressionPtr ScalarFnCollect::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    if (const auto binding = Resolve(expr.binding, analyses, projections)) {
        auto &[analysis, column_index] = *binding;
        // Column is used without function applied to it, register a conflict.
        analysis.col_to_fn[column_index] = nullptr;
    }
    return std::move(*ptr);
}

ExpressionPtr ScalarFnCollect::VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) {
    if (expr.children.size() != 1 ||
        expr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::move(*ptr);
    }
    const auto &bound_col = expr.children[0]->Cast<BoundColumnRefExpression>();

    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }
    auto &[analysis, column_index] = *binding;
    if (auto it = analysis.col_to_fn.find(column_index); it == analysis.col_to_fn.end()) {
        // This is the first time we see the column used by a single function.
        analysis.col_to_fn.emplace(column_index, &expr);
    } else if (it->second != &expr) {
        // Either column is used with different function in "expr" or
        // it->second is nullptr which indicates an existing conflict.
        it->second = nullptr;
    }

    // We don't want to descend into child BoundColumnRefExpression because we
    // have already registered a conflict if it was present.
    return std::move(*ptr);
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundColumnRefExpression &expr, ExpressionPtr *ptr) {
    if (const auto binding = Resolve(expr.binding, analyses, projections)) {
        const auto &[analysis, column_index] = *binding;

        // If we resolved this column, ScalarFnCollect has already processed it,
        // and expr_map either holds a valid pointer or a nullptr if there's a
        // conflict.
        D_ASSERT(analysis.col_to_fn.find(column_index) != analysis.col_to_fn.end());
        if (analysis.col_to_fn[column_index] == nullptr) {
            // This column has a conflict, don't replace it
            return std::move(*ptr);
        }

        // We updated GET's return type in TryPushdownScalarFunctions
        expr.return_type = analysis.get.types[column_index];
    }
    return std::move(*ptr);
}

ExpressionPtr ScalarFnReplace::VisitReplace(BoundFunctionExpression &expr, ExpressionPtr *ptr) {
    if (expr.children.size() != 1 ||
        expr.children[0]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
        return std::move(*ptr);
    }
    ExpressionPtr &bound_col_base = expr.children[0];
    const auto &bound_col = bound_col_base->Cast<BoundColumnRefExpression>();
    const auto binding = Resolve(bound_col.binding, analyses, projections);
    if (!binding) {
        return std::move(*ptr);
    }
    const auto &[analysis, column_index] = *binding;

    // If we resolved this column, ScalarFnCollect has already processed it,
    // and expr_map either holds a valid pointer or a nullptr if there's a
    // conflict.
    D_ASSERT(analysis.col_to_fn.find(column_index) != analysis.col_to_fn.end());
    if (analysis.col_to_fn[column_index] == nullptr) {
        // This column has a conflict, don't replace it
        return std::move(*ptr);
    }

    const TableColumnStorageIndex storage_index = analysis.storageIndex(column_index);
    std::cout << StringUtil::Format("Given original binding %s, replaced column with storage index %d, name "
                                    "%s in VORTEX get return type %s -> %s\n",
                                    bound_col.binding.ToString(),
                                    storage_index,
                                    analysis.get.names[storage_index],
                                    EnumUtil::ToString(bound_col_base->return_type.id()),
                                    EnumUtil::ToString(analysis.get.returned_types[storage_index].id()));

    // We updated GET's return type in TryPushdownScalarFunctions
    bound_col_base->return_type = analysis.get.types[column_index];

    return std::move(bound_col_base);
}

TableColumnStorageIndex GetAnalysis::storageIndex(TableColumnScanIndex column_index) const {
    return get.GetColumnIds()[column_index].GetPrimaryIndex();
}
