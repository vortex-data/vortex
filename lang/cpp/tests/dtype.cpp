// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex/dtype.hpp"
#include <catch2/catch_test_macros.hpp>
#include <vortex/data_source.hpp>

using namespace vortex;

namespace {
using enum vortex::PType;

TEST_CASE("Null dtype", "[dtype]") {
    auto d = dtype::null();
    REQUIRE(d.variant() == DataTypeVariant::Null);
    REQUIRE(d.nullable());
}

TEST_CASE("Decimal dtype", "[dtype]") {
    auto d = dtype::decimal(5, 2, false);
    REQUIRE(d.variant() == DataTypeVariant::Decimal);
    REQUIRE(d.decimal_precision() == 5);
    REQUIRE(d.decimal_scale() == 2);
    REQUIRE_FALSE(d.nullable());

    REQUIRE_THROWS_AS(d.fields(), VortexException);
    REQUIRE_THROWS_AS(d.list_element(), VortexException);
}

TEST_CASE("copy dtype", "[dtype]") {
    auto d = dtype::int32(true);
    DataType d2 = d;
    REQUIRE(d2.variant() == DataTypeVariant::Primitive);
    REQUIRE(d2.primitive_type() == I32);
    REQUIRE(d2.nullable());
    REQUIRE(d.variant() == DataTypeVariant::Primitive);
}

TEST_CASE("list dtype", "[dtype]") {
    auto d = dtype::list(dtype::float64(), dtype::Nullable);
    REQUIRE(d.variant() == DataTypeVariant::List);
    REQUIRE(d.nullable());
    REQUIRE(d.list_element().primitive_type() == F64);

    auto fsl = dtype::fixed_size_list(dtype::int16(), 4);
    REQUIRE(fsl.variant() == DataTypeVariant::FixedSizeList);
    REQUIRE(fsl.fixed_size_list_size() == 4);
    REQUIRE(fsl.fixed_size_list_element().primitive_type() == I16);
}

TEST_CASE("Struct DataType", "[dtype]") {
    DataType d = dtype::struct_({
        {"col1", dtype::uint8()},
        {"col2", dtype::binary(dtype::Nullable)},
    });

    REQUIRE(d.variant() == DataTypeVariant::Struct);
    REQUIRE_FALSE(d.nullable());
    const std::vector<StructField> fields = d.fields();
    REQUIRE(fields.size() == 2);
    REQUIRE(fields[0].name == "col1");
    REQUIRE(fields[1].name == "col2");
    REQUIRE(fields[0].dtype.primitive_type() == U8);
    REQUIRE(fields[1].dtype.variant() == DataTypeVariant::Binary);
    REQUIRE(fields[1].dtype.nullable());

    std::vector<StructField> fields_vec;
    fields_vec.emplace_back("col1", dtype::uint8());
    fields_vec.emplace_back("col2", dtype::utf8());
    d = dtype::struct_(fields_vec, dtype::Nullable);

    REQUIRE(d.variant() == DataTypeVariant::Struct);
    REQUIRE(d.nullable());
    const std::vector<StructField> built = d.fields();
    REQUIRE(built.size() == 2);
    REQUIRE(built[1].name == "col2");

    fields_vec = {};
    fields_vec.emplace_back("\xFF\xFE", dtype::uint8());
    REQUIRE_THROWS_AS(dtype::struct_(fields_vec), VortexException);
}
} // namespace
