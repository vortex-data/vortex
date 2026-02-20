// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/types/value.hpp"

#include "duckdb_vx.h"

extern "C" duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty) {
    const auto logical_type = reinterpret_cast<duckdb::LogicalType *>(ty);
    auto value = duckdb::make_uniq<duckdb::Value>(*logical_type);
    return reinterpret_cast<duckdb_value>(value.release());
}
