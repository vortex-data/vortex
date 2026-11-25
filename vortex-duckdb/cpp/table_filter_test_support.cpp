// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TEST-ONLY FFI constructors for DuckDB TableFilter objects (excluded from release builds).

#include "table_filter_test_support.h"

#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/bloom_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"

using namespace duckdb;

namespace {
struct TestDynamicFilterHandle {
    shared_ptr<DynamicFilterData> data;

    explicit TestDynamicFilterHandle(shared_ptr<DynamicFilterData> d) : data(std::move(d)) {
    }
};
} // namespace

extern "C" void duckdb_vx_test_table_filter_destroy(duckdb_vx_table_filter *ffi_filter) {
    if (!ffi_filter || !*ffi_filter) {
        return;
    }
    delete reinterpret_cast<TableFilter *>(*ffi_filter);
    *ffi_filter = nullptr;
}

extern "C" void duckdb_vx_test_dynamic_filter_data_destroy(duckdb_vx_test_dynamic_filter_data *ffi_data) {
    if (!ffi_data || !*ffi_data) {
        return;
    }
    delete reinterpret_cast<TestDynamicFilterHandle *>(*ffi_data);
    *ffi_data = nullptr;
}

extern "C" void duckdb_vx_test_dynamic_filter_data_set_value(duckdb_vx_test_dynamic_filter_data ffi_data,
                                                             duckdb_value value) {
    D_ASSERT(ffi_data);
    D_ASSERT(value);
    auto *handle = reinterpret_cast<TestDynamicFilterHandle *>(ffi_data);
    handle->data->SetValue(*reinterpret_cast<Value *>(value));
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_constant(duckdb_vx_expr_type comparison_type, duckdb_value value) {
    D_ASSERT(value);
    auto &val = *reinterpret_cast<Value *>(value);
    auto filter = make_uniq<LegacyConstantFilter>(static_cast<ExpressionType>(comparison_type), val);
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter duckdb_vx_test_table_filter_create_is_null(void) {
    auto filter = make_uniq<LegacyIsNullFilter>();
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter duckdb_vx_test_table_filter_create_is_not_null(void) {
    auto filter = make_uniq<LegacyIsNotNullFilter>();
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_conjunction_and(duckdb_vx_table_filter *children, size_t count) {
    auto filter = make_uniq<LegacyConjunctionAndFilter>();
    filter->child_filters.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        filter->child_filters.emplace_back(reinterpret_cast<TableFilter *>(children[i]));
    }
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_conjunction_or(duckdb_vx_table_filter *children, size_t count) {
    auto filter = make_uniq<LegacyConjunctionOrFilter>();
    filter->child_filters.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        filter->child_filters.emplace_back(reinterpret_cast<TableFilter *>(children[i]));
    }
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_struct_extract(idx_t child_idx,
                                                  const char *child_name,
                                                  duckdb_vx_table_filter child_filter) {
    D_ASSERT(child_name);
    D_ASSERT(child_filter);
    unique_ptr<TableFilter> child(reinterpret_cast<TableFilter *>(child_filter));
    auto filter = make_uniq<LegacyStructFilter>(child_idx, Identifier(child_name), std::move(child));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_optional(duckdb_vx_table_filter child_filter) {
    D_ASSERT(child_filter);
    unique_ptr<TableFilter> child(reinterpret_cast<TableFilter *>(child_filter));
    auto filter = make_uniq<LegacyOptionalFilter>(std::move(child));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter duckdb_vx_test_table_filter_create_in(duckdb_value *values, size_t count) {
    vector<Value> vals;
    vals.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        vals.push_back(*reinterpret_cast<Value *>(values[i]));
    }
    auto filter = make_uniq<LegacyInFilter>(std::move(vals));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_dynamic(duckdb_vx_expr_type comparison_type,
                                           duckdb_value initial_value,
                                           duckdb_vx_test_dynamic_filter_data *data_out) {
    D_ASSERT(initial_value);
    auto &val = *reinterpret_cast<Value *>(initial_value);
    auto data = make_shared_ptr<DynamicFilterData>(static_cast<ExpressionType>(comparison_type), val);
    if (data_out) {
        auto handle = make_uniq<TestDynamicFilterHandle>(data);
        *data_out = reinterpret_cast<duckdb_vx_test_dynamic_filter_data>(handle.release());
    }
    auto filter = make_uniq<LegacyDynamicFilter>(std::move(data));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter duckdb_vx_test_table_filter_create_bloom(void) {
    auto filter = make_uniq<LegacyBFTableFilter>(nullptr, false, "k", LogicalType::INTEGER);
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_expr_comparison(duckdb_vx_expr_type comparison_type,
                                                   duckdb_value value,
                                                   bool wrap_optional) {
    D_ASSERT(value);
    auto &val = *reinterpret_cast<Value *>(value);
    auto filter = ExpressionFilter::CreateComparisonFilter(static_cast<ExpressionType>(comparison_type), val);
    if (wrap_optional) {
        auto wrapped = CreateOptionalFilterExpression(std::move(filter->expr), val.type());
        filter = make_uniq<ExpressionFilter>(std::move(wrapped));
    }
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter duckdb_vx_test_table_filter_create_expr_in(duckdb_logical_type col_type,
                                                                             duckdb_value *values,
                                                                             size_t count,
                                                                             bool wrap_optional) {
    D_ASSERT(col_type);
    const auto &logical_type = *reinterpret_cast<LogicalType *>(col_type);
    unique_ptr<Expression> expr;
    if (count == 0) {
        auto op = make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN, LogicalType::BOOLEAN);
        op->GetChildrenMutable().push_back(make_uniq<BoundReferenceExpression>(logical_type, 0));
        expr = std::move(op);
    } else {
        vector<Value> vals;
        vals.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            vals.push_back(*reinterpret_cast<Value *>(values[i]));
        }
        expr = ExpressionFilter::CreateInExpression(make_uniq<BoundReferenceExpression>(logical_type, 0),
                                                    std::move(vals));
    }
    if (wrap_optional) {
        expr = CreateOptionalFilterExpression(std::move(expr), logical_type);
    }
    auto filter = make_uniq<ExpressionFilter>(std::move(expr));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter
duckdb_vx_test_table_filter_create_expr_dynamic(duckdb_vx_expr_type comparison_type,
                                                duckdb_value initial_value,
                                                bool wrap_optional,
                                                duckdb_vx_test_dynamic_filter_data *data_out) {
    D_ASSERT(initial_value);
    auto &val = *reinterpret_cast<Value *>(initial_value);
    auto data = make_shared_ptr<DynamicFilterData>(static_cast<ExpressionType>(comparison_type), val);
    if (data_out) {
        auto handle = make_uniq<TestDynamicFilterHandle>(data);
        *data_out = reinterpret_cast<duckdb_vx_test_dynamic_filter_data>(handle.release());
    }
    auto expr = CreateDynamicFilterExpression(std::move(data), val.type());
    if (wrap_optional) {
        expr = CreateOptionalFilterExpression(std::move(expr), val.type());
    }
    auto filter = make_uniq<ExpressionFilter>(std::move(expr));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}

extern "C" duckdb_vx_table_filter duckdb_vx_test_table_filter_create_expr_unsupported(void) {
    auto op = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, LogicalType::BOOLEAN);
    op->GetChildrenMutable().push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
    auto filter = make_uniq<ExpressionFilter>(std::move(op));
    return reinterpret_cast<duckdb_vx_table_filter>(filter.release());
}
