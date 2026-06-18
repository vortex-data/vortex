// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"

#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"

using namespace duckdb;

extern "C" idx_t duckdb_vx_table_filter_set_get(duckdb_vx_table_filter_set ffi_filter_set,
                                                size_t index,
                                                duckdb_vx_table_filter *table_filter_out) {
    auto &filters = reinterpret_cast<TableFilterSet *>(ffi_filter_set)->filters;
    if (filters.size() <= index) {
        *table_filter_out = nullptr;
        return 0;
    }
    auto &[column_idx, filter] = *std::next(filters.begin(), index);
    *table_filter_out = reinterpret_cast<duckdb_vx_table_filter>(filter.get());
    return column_idx;
}

extern "C" idx_t duckdb_vx_table_filter_set_size(duckdb_vx_table_filter_set ffi_filter_set) {
    auto filter_set = reinterpret_cast<TableFilterSet *>(ffi_filter_set);
    return filter_set->filters.size();
}

extern "C" duckdb_vx_table_filter_type duckdb_vx_table_filter_get_type(duckdb_vx_table_filter ffi_filter) {
    auto filter = reinterpret_cast<TableFilter *>(ffi_filter);
    return static_cast<duckdb_vx_table_filter_type>(filter->filter_type);
}

extern "C" const char *duckdb_vx_table_filter_to_debug_string(duckdb_vx_table_filter ffi_filter) {
    auto filter = reinterpret_cast<TableFilter *>(ffi_filter);
    auto str = filter->DebugToString();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

extern "C" void duckdb_vx_table_filter_get_constant(duckdb_vx_table_filter ffi_filter,
                                                    duckdb_vx_table_filter_constant *out) {
    if (!ffi_filter || !out) {
        return;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<ConstantFilter>();
    out->value = reinterpret_cast<duckdb_value>(&filter.constant);
    out->comparison_type = static_cast<duckdb_vx_expr_type>(filter.comparison_type);
}

extern "C" void duckdb_vx_table_filter_get_conjunction_or(duckdb_vx_table_filter ffi_filter,
                                                          duckdb_vx_table_filter_conjunction *out) {
    if (!ffi_filter || !out) {
        return;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<ConjunctionOrFilter>();
    out->children = reinterpret_cast<duckdb_vx_table_filter *>(filter.child_filters.data());
    out->children_count = filter.child_filters.size();
}

extern "C" void duckdb_vx_table_filter_get_conjunction_and(duckdb_vx_table_filter ffi_filter,
                                                           duckdb_vx_table_filter_conjunction *out) {
    if (!ffi_filter || !out) {
        return;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<ConjunctionAndFilter>();
    out->children = reinterpret_cast<duckdb_vx_table_filter *>(filter.child_filters.data());
    out->children_count = filter.child_filters.size();
}

// Wrapper to hold the shared pointer for dynamic filter data.
struct DynamicFilterDataWrapper {
    shared_ptr<DynamicFilterData> data;

    explicit DynamicFilterDataWrapper(shared_ptr<DynamicFilterData> d) : data(std::move(d)) {
    }
};

extern "C" void duckdb_vx_table_filter_get_dynamic(duckdb_vx_table_filter ffi_filter,
                                                   duckdb_vx_table_filter_dynamic *out) {
    if (!ffi_filter || !out) {
        return;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<DynamicFilter>();

    // Hold the lock while accessing the filter data.
    std::lock_guard<std::mutex> lock(filter.filter_data->lock);

    auto data_wrapper = duckdb::make_uniq<DynamicFilterDataWrapper>(filter.filter_data);
    out->data = reinterpret_cast<duckdb_vx_dynamic_filter_data>(data_wrapper.release());
    out->comparison_type = static_cast<duckdb_vx_expr_type>(filter.filter_data->filter->comparison_type);
}

extern "C" void duckdb_vx_dynamic_filter_data_free(duckdb_vx_dynamic_filter_data *ffi_data) {
    if (!ffi_data || !*ffi_data) {
        return;
    }
    delete reinterpret_cast<DynamicFilterDataWrapper *>(*ffi_data);
    *ffi_data = nullptr;
}

extern "C" duckdb_value duckdb_vx_dynamic_filter_data_get_value(duckdb_vx_dynamic_filter_data ffi_data) {
    if (!ffi_data) {
        return nullptr;
    }
    auto data_wrapper = reinterpret_cast<DynamicFilterDataWrapper *>(ffi_data);
    if (!data_wrapper->data) {
        return nullptr;
    }

    // Hold the lock while accessing the filter data.
    std::lock_guard<std::mutex> lock(data_wrapper->data->lock);

    if (!data_wrapper->data->filter || !data_wrapper->data->initialized) {
        return nullptr;
    }

    // Return a heap allocated copy of the value.
    return reinterpret_cast<duckdb_value>(new Value(data_wrapper->data->filter->constant));
}

extern "C" duckdb_vx_table_filter duckdb_vx_table_filter_get_optional(duckdb_vx_table_filter ffi_filter) {
    if (!ffi_filter) {
        return nullptr;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<OptionalFilter>();
    return reinterpret_cast<duckdb_vx_table_filter>(filter.child_filter.get());
}

extern "C" duckdb_vx_expr duckdb_vx_table_filter_get_expression(duckdb_vx_table_filter ffi_filter) {
    if (!ffi_filter) {
        return nullptr;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<ExpressionFilter>();
    return reinterpret_cast<duckdb_vx_expr>(filter.expr.get());
}

extern "C" void duckdb_vx_table_filter_get_struct_extract(duckdb_vx_table_filter ffi_filter,
                                                          duckdb_vx_table_filter_struct_extract *out) {
    if (!ffi_filter || !out) {
        return;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<StructFilter>();

    out->child_filter = reinterpret_cast<duckdb_vx_table_filter>(filter.child_filter.get());
    out->child_name_len = filter.child_name.size();
    out->child_name = static_cast<char *>(filter.child_name.data());
}

extern "C" void duckdb_vx_table_filter_get_in_filter(duckdb_vx_table_filter ffi_filter,
                                                     duckdb_vx_table_filter_in_filter *out) {
    if (!ffi_filter || !out) {
        return;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<InFilter>();

    out->values_count = filter.values.size();
    out->values = reinterpret_cast<duckdb_vx_values_vec>(&filter.values);
}

extern "C" duckdb_value duckdb_vx_values_vec_get(duckdb_vx_values_vec ffi_vec, size_t idx) {
    if (!ffi_vec) {
        return nullptr;
    }
    auto vec = reinterpret_cast<vector<Value> *>(ffi_vec);
    if (idx >= vec->size()) {
        return nullptr;
    }
    return reinterpret_cast<duckdb_value>(&(*vec)[idx]);
}
