// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cstdint>
#include <vortex/error.hpp>
#include <vortex/array.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace vortex;
using namespace vortex::expr::ops;

namespace {
using enum vortex::PType;
using enum ValidityType;

TEST_CASE("Null array", "[array]") {
    Array a = Array::null(1999);
    REQUIRE(a.size() == 1999);
    REQUIRE(a.nullable());
    REQUIRE(a.has_dtype(DataTypeVariant::Null));
}

TEST_CASE("Empty array", "[array]") {
    Session session;

    auto empty = Array::primitive<int32_t>({});
    REQUIRE(empty.size() == 0);
    REQUIRE(empty.is_primitive(I32));

    REQUIRE(empty.null_count() == 0);

    auto view = empty.values<int32_t>(session);
    REQUIRE(view.size() == 0);
    auto values = view.values();
    REQUIRE(values.empty());
}

void test_primitive_array(Array array, const int32_t *begin) {
    Session session;
    REQUIRE(array.size() == 3);
    REQUIRE(array.is_primitive(I32));
    REQUIRE_FALSE(array.nullable());
    REQUIRE(array.null_count() == 0);

    auto view = array.values<int32_t>(session);
    REQUIRE(view.size() == 3);
    REQUIRE(std::equal(view.values().begin(), view.values().end(), begin));
    REQUIRE_FALSE(view.is_null(1));
}

TEST_CASE("Primitive array", "[array]") {
    int32_t c_array[3] = {10, 20, 30};
    test_primitive_array(Array::primitive<int32_t>(c_array), c_array);

    const int32_t const_c_array[3] = {10, 20, 30};
    test_primitive_array(Array::primitive<int32_t>(const_c_array), const_c_array);

    const std::array<int32_t, 3> cpp_array = {10, 20, 30};
    test_primitive_array(Array::primitive<int32_t>(cpp_array), cpp_array.begin());

    std::vector<int32_t> cpp_vector = {10, 20, 30};
    test_primitive_array(Array::primitive<int32_t>(cpp_vector), cpp_vector.data());
}

TEST_CASE("values<T> with wrong type", "[array]") {
    Session session;
    std::vector<int32_t> data = {1};
    Array a = Array::primitive<int32_t>(data);
    REQUIRE_THROWS_AS(a.values<uint32_t>(session), VortexException);
}

TEST_CASE("Validity from a boolean mask", "[array]") {
    Session session;
    std::vector<int32_t> data = {10, 20, 30};
    std::vector<uint8_t> mask_bytes = {1, 0, 1};

    Array mask_u8 = Array::primitive<uint8_t>(std::span<const uint8_t>(mask_bytes));
    Array mask = mask_u8.apply(expr::root() == expr::lit<uint8_t>(1));

    Array a = Array::primitive<int32_t>(std::span<const int32_t>(data), Validity::from_array(mask));
    REQUIRE(a.nullable());
    REQUIRE(a.null_count() == 1);

    auto view = a.values<int32_t>(session);
    REQUIRE_FALSE(view.is_null(0));
    REQUIRE(view.is_null(1));
    REQUIRE_FALSE(view.is_null(2));
    REQUIRE(view.values()[0] == 10);
    REQUIRE(view.values()[2] == 30);

    Validity validity = a.validity();
    REQUIRE(validity.type() == FromArray);
    REQUIRE(validity.array().size() == 3);
}

TEST_CASE("Invalid validity", "[array]") {
    std::vector<uint8_t> invalid_mask = {1, 2, 3};
    Array mask = Array::primitive<uint8_t>(invalid_mask);
    REQUIRE_THROWS_AS(Validity::from_array(mask), VortexException);
}

TEST_CASE("AllInvalid", "[array]") {
    Session session;
    std::vector<int64_t> data = {1, 2};
    Array a = Array::primitive<int64_t>(std::span<const int64_t>(data), AllInvalid);
    REQUIRE(a.null_count() == 2);
    auto view = a.values<int64_t>(session);
    REQUIRE(view.is_null(0));
    REQUIRE(view.is_null(1));
}

TEST_CASE("make_struct and fields", "[array]") {
    Array empty = make_struct({});
    REQUIRE(empty.size() == 0);
    REQUIRE(empty.has_dtype(DataTypeVariant::Struct));
    REQUIRE(empty.dtype().fields().size() == 0);

    std::vector<uint8_t> ages = {10, 20, 30};
    std::vector<uint16_t> heights = {150, 160, 170};

    Array s = make_struct({
        {"age", Array::primitive<uint8_t>(ages)},
        {"height", Array::primitive<uint16_t>(heights, AllValid)},
    });

    REQUIRE(s.size() == 3);
    REQUIRE(s.has_dtype(DataTypeVariant::Struct));
    REQUIRE(s.dtype().fields().size() == 2);

    Array by_index = s.field(0);
    REQUIRE(by_index.is_primitive(U8));

    Session session;
    Array by_name = s.field("height");
    REQUIRE(by_name.is_primitive(U16));
    auto view = by_name.values<uint16_t>(session);
    REQUIRE(view.values()[2] == 170);

    REQUIRE_THROWS_AS(s.field(2), VortexException);
    REQUIRE_THROWS_AS(s.field("nope"), VortexException);

    std::vector<ColumnField> fields_vec;
    fields_vec.emplace_back("age", Array::primitive<uint8_t>(ages));
    Array other = make_struct(fields_vec);
    REQUIRE(other.size() == 3);
    REQUIRE(other.has_dtype(DataTypeVariant::Struct));
}

TEST_CASE("Mismatched field length", "[array]") {
    std::vector<uint8_t> a = {1, 2};
    std::vector<uint8_t> b = {1, 2, 3};
    REQUIRE_THROWS_AS(make_struct({
                          {"a", Array::primitive<uint8_t>(a)},
                          {"b", Array::primitive<uint8_t>(b)},
                      }),
                      VortexException);
}

TEST_CASE("Slice", "[array]") {
    Session session;
    std::vector<int16_t> data = {0, 1, 2, 3, 4, 5};
    Array a = Array::primitive<int16_t>(data);
    Array sliced = a.slice(2, 5);
    REQUIRE(sliced.size() == 3);
    auto view = sliced.values<int16_t>(session);
    REQUIRE(view.values()[0] == 2);
    REQUIRE(view.values()[2] == 4);

    REQUIRE_THROWS_AS(a.slice(2, 100), VortexException);
}

TEST_CASE("Bool array", "[array]") {
    Session session;

    constexpr size_t ELEMENTS = 9;
    std::array<bool, ELEMENTS> data = {true, false, true, true, false, true, true, false, true};
    std::vector<uint8_t> bitpacked(2);
    for (size_t i = 0; i < ELEMENTS; ++i) {
        if (data[i]) {
            bitpacked[i / 8] |= 1 << (i % 8);
        }
    }

    BoolView bool_view = {bitpacked, ELEMENTS};
    Array array = Array::bool_array(bool_view);
    REQUIRE(array.size() == data.size());
    REQUIRE(array.has_dtype(DataTypeVariant::Bool));
    REQUIRE_FALSE(array.nullable());

    auto view = array.bools(session);
    BoolView values = view.values();
    for (size_t i = 0; i < values.elements(); ++i) {
        REQUIRE(values[i] == data[i]);
    }

    array = array.slice(3, array.size());
    view = array.bools(session);
    values = view.values();

    Array roundtrip = Array::bool_array(values);
    REQUIRE(roundtrip.size() == view.size());
    REQUIRE_FALSE(roundtrip.nullable());

    auto roundtrip_view = roundtrip.bools(session);
    BoolView roundtrip_values = roundtrip_view.values();
    for (size_t i = 0; i < view.size(); ++i) {
        REQUIRE(roundtrip_values[i] == values[i]);
        REQUIRE(roundtrip_values[i] == data[i + 3]);
    }
}

TEST_CASE("Error with a code", "[array]") {
    std::vector<int16_t> data = {0};
    Array a = Array::primitive<int16_t>(data);
    try {
        (void)a.slice(2, 100);
        FAIL("expected exception");
    } catch (const VortexException &e) {
        REQUIRE_FALSE(std::string(e.what()).empty());
        (void)e.code();
    }
}
} // namespace
