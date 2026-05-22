// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/types/value.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx.h"

extern "C" duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty) {
    const auto logical_type = reinterpret_cast<duckdb::LogicalType *>(ty);
    auto value = duckdb::make_uniq<duckdb::Value>(*logical_type);
    return reinterpret_cast<duckdb_value>(value.release());
}

extern "C" duckdb_value duckdb_vx_value_clone(duckdb_value value) {
    if (!value) {
        return nullptr;
    }

    const auto ddb_value = reinterpret_cast<duckdb::Value *>(value);
    return reinterpret_cast<duckdb_value>(new duckdb::Value(*ddb_value));
}
