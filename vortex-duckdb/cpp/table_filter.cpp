#include "duckdb/planner/table_filter.hpp"
#include "duckdb_vx.h"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"

using namespace duckdb;

extern "C" idx_t duckdb_vx_table_filter_set_get(duckdb_vx_table_filter_set ffi_filter_set, size_t index,
                                                duckdb_vx_table_filter *table_filter_out) {
    auto filter_set = reinterpret_cast<TableFilterSet *>(ffi_filter_set);
    auto it = std::next(filter_set->filters.begin(), index);

    if (it == filter_set->filters.end()) {
        table_filter_out = nullptr;
        return 0;
    }

    *table_filter_out = reinterpret_cast<duckdb_vx_table_filter>(it->second.get());
    return it->first;
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

extern "C" duckdb_vx_dynamic_filter_data
duckdb_vx_table_filter_get_dynamic(duckdb_vx_table_filter ffi_filter) {
    if (!ffi_filter) {
        return nullptr;
    }
    auto &filter = reinterpret_cast<TableFilter *>(ffi_filter)->Cast<DynamicFilter>();
    return reinterpret_cast<duckdb_vx_dynamic_filter_data>(filter.filter_data.get());
}
