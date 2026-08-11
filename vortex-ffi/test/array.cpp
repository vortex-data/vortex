// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <vortex.h>
#include "common.h"

TEST_CASE("Null array creation", "[array]") {
    const vx_array *array = vx_array_new_null(1999);
    REQUIRE(array != nullptr);
    REQUIRE(vx_array_is_nullable(array));
    REQUIRE(vx_array_has_dtype(array, DTYPE_NULL));
    const vx_dtype *dtype = vx_array_dtype(array);
    defer {
        vx_dtype_free(dtype);
    };
    REQUIRE(vx_dtype_get_variant(dtype) == DTYPE_NULL);
    REQUIRE(vx_array_len(array) == 1999);
    vx_array_free(array);
}

TEST_CASE("Primitive array creation", "[array]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };

    std::vector<uint8_t> buffer(20, 1);
    buffer[3] = 8;

    vx_validity validity = {};
    validity.type = VX_VALIDITY_ALL_VALID;
    vx_error *error = nullptr;
    const vx_array *array = vx_array_new_primitive(PTYPE_U8, buffer.data(), buffer.size(), &validity, &error);

    require_no_error(error);
    REQUIRE(array != nullptr);
    REQUIRE(vx_array_has_dtype(array, DTYPE_PRIMITIVE));
    const vx_dtype *dtype = vx_array_dtype(array);
    REQUIRE(vx_dtype_get_variant(dtype) == DTYPE_PRIMITIVE);
    defer {
        vx_dtype_free(dtype);
    };
    REQUIRE(vx_array_is_primitive(array, PTYPE_U8));
    REQUIRE(vx_array_len(array) == buffer.size());

    for (size_t i = 0; i < buffer.size(); ++i) {
        REQUIRE(buffer[i] == array_get_u8(session, array, i));
    }

    buffer = {};

    for (size_t i = 0; i < 20; ++i) {
        REQUIRE(array_get_u8(session, array, i) == (i == 3 ? 8 : 1));
    }

    vx_array_free(array);
}

TEST_CASE("Bool view", "[array]") {
    std::vector<uint8_t> buffer(2, UINT8_MAX);
    constexpr size_t ELEMENTS = 10;
    constexpr size_t OFFSET = 6;
    vx_bool_view view {.ptr = buffer.data(), .elements = ELEMENTS, .bit_offset = OFFSET};

    REQUIRE(vx_bool_view_len(view) == 2);

    for (size_t i = 0; i < view.elements; ++i) {
        REQUIRE(vx_bool_view_nth(view, i) == 1);
    }

    // bit_offset is 6. buffer[0] holds elements 0-1. buffer[1] holds elements
    // 2-9. buffer[1] = 1 sets element 2 to 1 and elements 3-9 to 0.
    buffer[1] = 1;
    REQUIRE(vx_bool_view_nth(view, 0) == 1);
    REQUIRE(vx_bool_view_nth(view, 1) == 1);
    REQUIRE(vx_bool_view_nth(view, 2) == 1);
    for (size_t i = 3; i < ELEMENTS; ++i) {
        REQUIRE(vx_bool_view_nth(view, i) == 0);
    }
}

TEST_CASE("Bool array", "[array]") {
    const std::vector<bool> values = {true, true, false, true, false, true, true, false, true, true};
    std::vector<uint8_t> words(2, 0);
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i]) {
            words[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
    }
    vx_bool_view view {.ptr = words.data(), .elements = values.size(), .bit_offset = 0};

    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;
    vx_error *error = nullptr;
    const vx_array *array = vx_array_new_bool(&view, &validity, &error);
    require_no_error(error);
    defer {
        vx_array_free(array);
    };
    REQUIRE(vx_array_len(array) == values.size());

    for (size_t i = 0; i < values.size(); ++i) {
        REQUIRE(vx_array_get_bool(array, i) == values[i]);
    }

    view = vx_array_data_ptr_bool(array, &error);
    require_no_error(error);
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t bit = view.bit_offset + i;
        const bool actual = (view.ptr[bit / 8] >> (bit % 8)) & 1;
        REQUIRE(actual == values[i]);
    }
}

TEST_CASE("Array with validity", "[array]") {
    const std::vector<bool> valid = {true, false, true, true, false};
    std::vector<uint8_t> words(1, 0);
    for (size_t i = 0; i < valid.size(); ++i) {
        if (valid[i]) {
            words[0] |= static_cast<uint8_t>(1u << i);
        }
    }
    vx_bool_view validity_view {.ptr = words.data(), .elements = valid.size(), .bit_offset = 0};

    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;
    vx_error *error = nullptr;
    const vx_array *validity_array = vx_array_new_bool(&validity_view, &validity, &error);
    require_no_error(error);
    defer {
        vx_array_free(validity_array);
    };

    validity = {};
    validity.type = VX_VALIDITY_ARRAY;
    validity.array = validity_array;

    std::vector<uint8_t> data(valid.size(), 7);
    const vx_array *array = vx_array_new_primitive(PTYPE_U8, data.data(), data.size(), &validity, &error);
    require_no_error(error);
    defer {
        vx_array_free(array);
    };

    REQUIRE(vx_array_is_nullable(array));
    REQUIRE(vx_array_len(array) == valid.size());

    for (size_t i = 0; i < valid.size(); ++i) {
        REQUIRE(vx_array_element_is_invalid(array, i, &error) == !valid[i]);
        require_no_error(error);
    }
    REQUIRE(vx_array_invalid_count(array, &error) == 2);
    require_no_error(error);

    validity = {};
    vx_array_get_validity(array, &validity, &error);
    require_no_error(error);
    REQUIRE(validity.type == VX_VALIDITY_ARRAY);
    vx_array_free(validity.array);
}

TEST_CASE("Struct array creation", "[array]") {
    vx_error *error = nullptr;

    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;

    const vx_array *field_array = vx_array_new_null(5);
    CHECK(field_array != nullptr);
    vx_struct_column_builder *builder = vx_struct_column_builder_new(&validity, 2);
    CHECK(builder != nullptr);

    vx_struct_column_builder_add_field(builder, vx_view_from_cstr("age"), field_array, &error);
    vx_array_free(field_array);

    SECTION("Struct array builder free") {
        vx_struct_column_builder_free(builder);
    }

    SECTION("Struct array builder finalize") {
        const vx_array *struct_array = vx_struct_column_builder_finalize(builder, &error);
        vx_array_free(struct_array);
    }
}
