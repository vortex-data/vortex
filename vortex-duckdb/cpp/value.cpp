// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/variant.hpp"
#include "duckdb/function/scalar/variant_utils.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx.h"

using namespace duckdb;

extern "C" duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty) {
    const auto logical_type = reinterpret_cast<LogicalType *>(ty);
    auto value = make_uniq<Value>(*logical_type);
    return reinterpret_cast<duckdb_value>(value.release());
}

extern "C" duckdb_value
duckdb_vx_variant_value_unwrap(duckdb_value ffi_value, bool *outer_null, duckdb_vx_error *err) {
    try {
        auto value = reinterpret_cast<Value *>(ffi_value);
        if (!value || value->IsNull()) {
            *outer_null = true;
            *err = nullptr;
            return nullptr;
        }

        Vector vector(LogicalType::VARIANT(), 1);
        vector.SetValue(0, *value);

        RecursiveUnifiedVectorFormat format;
        Vector::RecursiveToUnifiedFormat(vector, format);
        UnifiedVariantVectorData variant(format);
        if (!variant.RowIsValid(0)) {
            *outer_null = true;
            *err = nullptr;
            return nullptr;
        }

        *outer_null = false;
        auto unwrapped = make_uniq<Value>(VariantUtils::ConvertVariantToValue(variant, 0, 0));
        *err = nullptr;
        return reinterpret_cast<duckdb_value>(unwrapped.release());
    } catch (std::exception &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
        return nullptr;
    }
}
