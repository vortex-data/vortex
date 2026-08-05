// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex/error.hpp"
#include <catch2/catch_test_macros.hpp>
#include <string_view>
#include <vortex/scalar.hpp>

using namespace vortex;
using namespace std::string_view_literals;

namespace {
using enum vortex::PType;
using scalar::decimal;
using scalar::of;

TEST_CASE("Boolean scalar", "[scalar]") {
    Scalar s = scalar::of(true);
    REQUIRE_FALSE(s.is_null());
    REQUIRE(s.dtype().variant() == DataTypeVariant::Bool);
}

TEST_CASE("Integer scalars", "[scalar]") {
    REQUIRE(scalar::of<uint8_t>(42).dtype().primitive_type() == U8);
    REQUIRE(scalar::of<uint16_t>(42).dtype().primitive_type() == U16);
    REQUIRE(scalar::of<uint32_t>(42).dtype().primitive_type() == U32);
    REQUIRE(scalar::of<uint64_t>(42).dtype().primitive_type() == U64);
    REQUIRE(scalar::of<int8_t>(-1).dtype().primitive_type() == I8);
    REQUIRE(scalar::of<int16_t>(-1).dtype().primitive_type() == I16);
    REQUIRE(scalar::of<int32_t>(-1).dtype().primitive_type() == I32);
    REQUIRE(scalar::of<int64_t>(-1).dtype().primitive_type() == I64);
}

TEST_CASE("Float scalars", "[scalar]") {
    REQUIRE(scalar::of(1.5f).dtype().primitive_type() == F32);
    REQUIRE(scalar::of(1.5).dtype().primitive_type() == F64);
    REQUIRE(scalar::of(float16_t {0x3C00}).dtype().primitive_type() == F16);
}

TEST_CASE("Nullable scalars", "[scalar]") {
    Scalar s = scalar::of<int32_t>(0, true);
    REQUIRE(s.dtype().nullable());
    REQUIRE_FALSE(s.is_null());
}

TEST_CASE("Null scalar", "[scalar]") {
    Scalar s = scalar::null(dtype::int32(true));
    REQUIRE(s.is_null());
    REQUIRE(s.dtype().variant() == DataTypeVariant::Primitive);
    REQUIRE_THROWS_AS(s.get<int32_t>(), VortexException);
}

TEST_CASE("UTF-8 scalar", "[scalar]") {
    Scalar s = scalar::of("hello"sv);
    REQUIRE_FALSE(s.is_null());
    REQUIRE(s.dtype().variant() == DataTypeVariant::Utf8);
    REQUIRE_THROWS_AS(s.get<float>(), VortexException);

    REQUIRE_FALSE(scalar::of(""sv).is_null());
    s = scalar::of("Широкая строка"sv);
    REQUIRE_THROWS_AS(s.get<bool>(), VortexException);
    REQUIRE_THROWS_AS(s.get_decimal<int8_t>(), VortexException);
    REQUIRE(s.dtype().variant() == DataTypeVariant::Utf8);

    REQUIRE_THROWS_AS(scalar::of("\xFF\xFE"sv), VortexException);
}

TEST_CASE("Binary scalar", "[scalar]") {
    const std::byte bytes[] = {std::byte {1}, std::byte {2}, std::byte {0}, std::byte {4}};
    Scalar s = scalar::of(std::span<const std::byte> {bytes});
    REQUIRE_FALSE(s.is_null());
    REQUIRE(s.dtype().variant() == DataTypeVariant::Binary);
}

TEST_CASE("Decimal scalars", "[scalar]") {
    Scalar d8 = scalar::decimal<int8_t>(56, 5, 2);
    REQUIRE(d8.dtype().variant() == DataTypeVariant::Decimal);
    REQUIRE(d8.dtype().decimal_precision() == 5);
    REQUIRE(d8.dtype().decimal_scale() == 2);

    Scalar d16 = scalar::decimal(int16_t(1234), 5, 2);
    REQUIRE(d16.dtype().variant() == DataTypeVariant::Decimal);
    REQUIRE(d16.dtype().decimal_precision() == 5);
    REQUIRE(d16.dtype().decimal_scale() == 2);

    Scalar d32 = scalar::decimal(int32_t(1234), 5, 2);
    REQUIRE(d32.dtype().variant() == DataTypeVariant::Decimal);
    REQUIRE(d32.dtype().decimal_precision() == 5);
    REQUIRE(d32.dtype().decimal_scale() == 2);

    Scalar d64 = scalar::decimal<int64_t>(99999, 12, 3);
    REQUIRE(d64.dtype().variant() == DataTypeVariant::Decimal);
    REQUIRE(d64.dtype().decimal_precision() == 12);
    REQUIRE(d64.dtype().decimal_scale() == 3);
}

TEST_CASE("Invalid scalar getter", "[scalar]") {
    Scalar scalar = of<int32_t>(0);
    REQUIRE_THROWS_AS(scalar.get<double>(), VortexException);
}

TEST_CASE("Primitive getters", "[scalar]") {
    REQUIRE(of<uint64_t>(1ULL << 40).get<uint64_t>() == (1ULL << 40));
    REQUIRE(of<int32_t>(-7).get<int32_t>() == -7);
    REQUIRE(of(2.5).get<double>() == 2.5);
    REQUIRE(of(true).get<bool>());
}

TEST_CASE("String getters", "[scalar]") {
    Scalar s = of("hello"sv);
    REQUIRE(s.get<std::string_view>() == "hello"sv);

    const std::byte bytes[] = {std::byte {1}, std::byte {2}, std::byte {0}, std::byte {4}};
    Scalar b = of(std::span<const std::byte> {bytes});
    BinaryView view = b.get<BinaryView>();
    REQUIRE(view.size() == 4);
    REQUIRE(view[0] == std::byte {1});
    REQUIRE(view[3] == std::byte {4});
}

TEST_CASE("Decimal getters", "[scalar]") {
    Scalar s = scalar::null(dtype::decimal(5, 2));
    REQUIRE_THROWS_AS(s.get_decimal<int8_t>(), VortexException);

    REQUIRE(decimal<int8_t>(56, 5, 2).get_decimal<int8_t>() == 56);
    REQUIRE(decimal<int16_t>(1234, 5, 2).get_decimal<int16_t>() == 1234);
    REQUIRE(decimal<int32_t>(5678, 6, 2).get_decimal<int32_t>() == 5678);
    REQUIRE(decimal<int64_t>(99999, 12, 3).get_decimal<int64_t>() == 99999);
}

TEST_CASE("Copy scalar", "[scalar]") {
    Scalar a = scalar::of<int64_t>(42);
    Scalar b = a;
    REQUIRE(a.dtype().primitive_type() == I64);
    REQUIRE(b.dtype().primitive_type() == I64);

    Scalar c = scalar::of<int32_t>(1);
    c = b;
    REQUIRE(c.dtype().primitive_type() == I64);
}
} // namespace
