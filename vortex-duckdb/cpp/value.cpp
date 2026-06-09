// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/types/geometry_crs.hpp"
#include "duckdb/common/types/value.hpp"

#include "duckdb_vx.h"

extern "C" duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty) {
    const auto logical_type = reinterpret_cast<duckdb::LogicalType *>(ty);
    auto value = duckdb::make_uniq<duckdb::Value>(*logical_type);
    return reinterpret_cast<duckdb_value>(value.release());
}

extern "C" duckdb_value duckdb_vx_value_create_geometry(const uint8_t *wkb, idx_t len, const char *crs) {
    const auto bytes = reinterpret_cast<duckdb::const_data_ptr_t>(wkb);
    auto value =
        (crs == nullptr || *crs == '\0')
            ? duckdb::Value::GEOMETRY(bytes, len)
            : duckdb::Value::GEOMETRY(bytes, len, duckdb::CoordinateReferenceSystem(std::string(crs)));
    auto owned = duckdb::make_uniq<duckdb::Value>(std::move(value));
    return reinterpret_cast<duckdb_value>(owned.release());
}

extern "C" duckdb_blob duckdb_vx_value_get_geometry(duckdb_value value) {
    if (value == nullptr) {
        return {nullptr, 0};
    }
    const auto val = reinterpret_cast<duckdb::Value *>(value);
    if (val->type().id() != duckdb::LogicalTypeId::GEOMETRY) {
        return {nullptr, 0};
    }
    const auto &str = duckdb::StringValue::Get(*val);
    const auto size = str.size();
    auto buf = reinterpret_cast<void *>(duckdb_malloc(size));
    if (size > 0) {
        memcpy(buf, str.c_str(), size);
    }
    return {buf, size};
}
