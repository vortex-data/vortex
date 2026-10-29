// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "table_filter.h"

#include "duckdb/planner/filter/constant_filter.hpp"

using namespace duckdb;

extern "C" duckdb_vx_table_filter duckdb_vx_test_constant_filter(duckdb_value value) noexcept {
    if (!value) {
        return nullptr;
    }

    try {
        auto &constant = *reinterpret_cast<Value *>(value);
        if (constant.IsNull()) {
            return nullptr;
        }
        auto filter = new LegacyConstantFilter(ExpressionType::COMPARE_EQUAL, constant);
        return reinterpret_cast<duckdb_vx_table_filter>(filter);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void duckdb_vx_test_destroy_constant_filter(duckdb_vx_table_filter filter) noexcept {
    delete reinterpret_cast<LegacyConstantFilter *>(filter);
}
