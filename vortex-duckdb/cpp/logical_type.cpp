// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/types.hpp"

#include "duckdb_vx.h"
#include "duckdb_vx/logical_type.h"

#include "duckdb/common/types.hpp"
#include <cassert>

duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty) {
    assert(ty != nullptr);
    auto *src = reinterpret_cast<duckdb::LogicalType *>(ty);

    // Leverage the C++ copy constructor.
    auto *copy = new duckdb::LogicalType(*src);
    return reinterpret_cast<duckdb_logical_type>(copy);
}

char *duckdb_vx_logical_type_stringify(duckdb_logical_type c_type) {
    auto type = reinterpret_cast<duckdb::LogicalType *>(c_type);
    auto str = type->ToString();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}
