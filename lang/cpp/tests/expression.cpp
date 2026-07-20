// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include <vortex/array.hpp>
#include <vortex/expression.hpp>
#include <vortex/session.hpp>

using namespace vortex;

namespace {
using enum vortex::PType;

TEST_CASE("Expression copy", "[expr]") {
    Expression clone = expr::root();
    {
        Expression orig = expr::col("age");
        clone = orig;
    }
    Expression another = clone;
    (void)another;
    SUCCEED();
}

TEST_CASE("Apply root()", "[expr]") {
    Session session;
    std::vector<int32_t> data = {10, 20, 30, 40, 50};
    Array array = Array::primitive<int32_t>(data);

    Array applied = array.apply(expr::root());

    REQUIRE(applied.size() == data.size());
    REQUIRE(applied.is_primitive(I32));
    auto values = applied.values<int32_t>(session);
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(values.values()[i] == data[i]);
    }
}

TEST_CASE("Apply projection", "[expr]") {
    Session session;
    std::vector<uint8_t> ages = {10, 20, 30};
    std::vector<uint16_t> heights = {150, 160, 170};

    Array struct_arr = make_struct({
        {"age", Array::primitive<uint8_t>(ages)},
        {"height", Array::primitive<uint16_t>(heights)},
    });

    Array projected = struct_arr.apply(expr::col("age"));
    REQUIRE(projected.size() == ages.size());
    REQUIRE(projected.is_primitive(U8));
    auto values = projected.values<uint8_t>(session);
    for (size_t i = 0; i < ages.size(); ++i) {
        REQUIRE(values.values()[i] == ages[i]);
    }
}

TEST_CASE("Apply arithmetic", "[expr]") {
    Session session;
    std::vector<int32_t> data = {1, 2, 3, 4, 5};
    Array array = Array::primitive<int32_t>(data);

    Expression e = expr::add(expr::root(), expr::lit<int32_t>(10));
    Array applied = array.apply(e);

    REQUIRE(applied.is_primitive(I32));
    auto values = applied.values<int32_t>(session);
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(values.values()[i] == data[i] + 10);
    }
}

TEST_CASE("Operator overloading", "[expr]") {
    using namespace vortex::expr::ops;
    Session session;
    std::vector<uint32_t> data = {1, 2, 2, 7};
    Array array = Array::primitive<uint32_t>(data);

    Array applied = array.apply(expr::root() == expr::lit<uint32_t>(2));
    auto bits = applied.bools(session);
    REQUIRE(bits.size() == data.size());
    REQUIRE_FALSE(bits.value(0));
    REQUIRE(bits.value(1));
    REQUIRE(bits.value(2));
    REQUIRE_FALSE(bits.value(3));
}

TEST_CASE("Apply error", "[expr]") {
    std::vector<uint8_t> data = {1, 2, 3};
    Array array = Array::primitive<uint8_t>(data);

    Expression bad = expr::add(expr::root(), expr::lit<int32_t>(1));
    REQUIRE_THROWS_AS(array.apply(bad), VortexException);
}

TEST_CASE("Empty conjunction", "[expr]") {
    REQUIRE_THROWS_AS(expr::and_all({}), VortexException);
    REQUIRE_THROWS_AS(expr::or_all({}), VortexException);
}
} // namespace
