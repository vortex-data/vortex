// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/logical_type.h"
#include "duckdb/common/types.hpp"
#include <cassert>
#include <string>

duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty) {
    D_ASSERT(ty);
    auto *src = reinterpret_cast<duckdb::LogicalType *>(ty);
    auto copy = duckdb::make_uniq<duckdb::LogicalType>(*src);
    return reinterpret_cast<duckdb_logical_type>(copy.release());
}

char *duckdb_vx_logical_type_stringify(duckdb_logical_type c_type) {
    auto type = reinterpret_cast<duckdb::LogicalType *>(c_type);
    auto str = type->ToString();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

duckdb_logical_type duckdb_vx_create_geometry(const char *crs) {
    D_ASSERT(crs);
    auto geom =
        (*crs == '\0') ? duckdb::LogicalType::GEOMETRY() : duckdb::LogicalType::GEOMETRY(std::string(crs));
    auto copy = duckdb::make_uniq<duckdb::LogicalType>(std::move(geom));
    return reinterpret_cast<duckdb_logical_type>(copy.release());
}
