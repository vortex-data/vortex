// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/duckdb_diagnostics.h"
#include "duckdb_vx/logical_type.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/types.hpp"
DUCKDB_INCLUDES_END
#include <cassert>

duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty) {
    D_ASSERT(ty);
    auto *src = reinterpret_cast<duckdb::LogicalType *>(ty);
    auto copy = duckdb::make_uniq<duckdb::LogicalType>(*src);
    return reinterpret_cast<duckdb_logical_type>(copy.release());
}

duckdb_logical_type duckdb_vx_logical_type_variant(void) {
    auto type = duckdb::make_uniq<duckdb::LogicalType>(duckdb::LogicalType::VARIANT());
    return reinterpret_cast<duckdb_logical_type>(type.release());
}

char *duckdb_vx_logical_type_stringify(duckdb_logical_type c_type) {
    auto type = reinterpret_cast<duckdb::LogicalType *>(c_type);
    auto str = type->ToString();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}
