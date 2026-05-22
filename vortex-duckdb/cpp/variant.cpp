// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/variant.h"
#include "duckdb_vx/error.hpp"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/variant_value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "reader/variant/variant_shredded_conversion.hpp"
#include "writer/variant_column_writer.hpp"
DUCKDB_INCLUDES_END

#include <exception>

using namespace duckdb;

namespace {

duckdb::LogicalType UnshreddedParquetVariantType() {
    duckdb::child_list_t<duckdb::LogicalType> children;
    children.emplace_back("metadata", duckdb::LogicalType::BLOB);
    children.emplace_back("value", duckdb::LogicalType::BLOB);
    auto type = duckdb::LogicalType::STRUCT(std::move(children));
    type.SetAlias("PARQUET_VARIANT");
    return type;
}

duckdb::LogicalType ParquetVariantGroupType(Vector &value, optional_ptr<Vector> typed_value) {
    duckdb::child_list_t<duckdb::LogicalType> children;
    children.emplace_back("value", value.GetType());
    if (typed_value) {
        children.emplace_back("typed_value", typed_value->GetType());
    }
    return duckdb::LogicalType::STRUCT(std::move(children));
}

void SetException(duckdb_vx_error *err, const std::exception &e) {
    vortex::SetError(err, e.what());
}

void SetUnknownException(duckdb_vx_error *err) {
    vortex::SetError(err, "unknown DuckDB Variant conversion error");
}

} // namespace

extern "C" duckdb_vector duckdb_vx_variant_to_parquet(duckdb_vector variant, idx_t len, duckdb_vx_error *err) {
    try {
        auto &variant_vector = *reinterpret_cast<Vector *>(variant);
        auto result_type = UnshreddedParquetVariantType();
        auto result = make_uniq<Vector>(result_type, len);

        DataChunk input;
        input.Initialize(Allocator::DefaultAllocator(), {duckdb::LogicalType::VARIANT()}, len);
        input.SetCardinality(len);
        input.data[0].Reference(variant_vector);

        auto transform = VariantColumnWriter::GetTransformFunction();
        ExpressionExecutorState root;
        BoundFunctionExpression expr(result_type, transform, {}, nullptr);
        ExpressionState state(expr, root);
        transform.function(input, state, *result);

        *err = nullptr;
        return reinterpret_cast<duckdb_vector>(result.release());
    } catch (const std::exception &e) {
        SetException(err, e);
    } catch (...) {
        SetUnknownException(err);
    }
    return nullptr;
}

extern "C" void duckdb_vx_variant_from_parquet(duckdb_vector metadata,
                                               duckdb_vector value,
                                               duckdb_vector typed_value,
                                               bool has_typed_value,
                                               duckdb_vector out,
                                               idx_t len,
                                               duckdb_vx_error *err) {
    try {
        auto &metadata_vector = *reinterpret_cast<Vector *>(metadata);
        auto &value_vector = *reinterpret_cast<Vector *>(value);
        optional_ptr<Vector> typed_value_vector;
        if (has_typed_value) {
            typed_value_vector = reinterpret_cast<Vector *>(typed_value);
        }

        Vector group(ParquetVariantGroupType(value_vector, typed_value_vector), len);
        auto &entries = StructVector::GetEntries(group);
        entries[0]->Reference(value_vector);
        if (typed_value_vector) {
            entries[1]->Reference(*typed_value_vector);
        }

        auto values = VariantShreddedConversion::Convert(metadata_vector, group, 0, len, len);
        auto &out_vector = *reinterpret_cast<Vector *>(out);
        VariantValue::ToVARIANT(values, out_vector);
        *err = nullptr;
    } catch (const std::exception &e) {
        SetException(err, e);
    } catch (...) {
        SetUnknownException(err);
    }
}
