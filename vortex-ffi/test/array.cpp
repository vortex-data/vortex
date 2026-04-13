// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <vortex.h>
#include "common.h"

TEST_CASE("Null array creation", "[array]") {
    const vx_array *array = vx_array_new_null(1999);
    REQUIRE(array != nullptr);
    REQUIRE(vx_array_is_nullable(array));
    REQUIRE(vx_array_has_dtype(array, DTYPE_NULL));
    REQUIRE(vx_dtype_get_variant(vx_array_dtype(array)) == DTYPE_NULL);
    REQUIRE(vx_array_len(array) == 1999);
    vx_array_free(array);
}

TEST_CASE("Primitive array creation", "[array]") {
    std::vector<uint8_t> buffer(20, 1);
    buffer[3] = 8;

    vx_validity validity = {};
    validity.type = VX_VALIDITY_ALL_VALID;
    vx_error *error = nullptr;
    const vx_array *array = vx_array_new_primitive(PTYPE_U8, buffer.data(), buffer.size(), &validity, &error);

    require_no_error(error);
    REQUIRE(array != nullptr);
    REQUIRE(vx_array_has_dtype(array, DTYPE_PRIMITIVE));
    REQUIRE(vx_dtype_get_variant(vx_array_dtype(array)) == DTYPE_PRIMITIVE);
    REQUIRE(vx_array_is_primitive(array, PTYPE_U8));
    REQUIRE(vx_array_len(array) == buffer.size());

    for (size_t i = 0; i < buffer.size(); ++i) {
        REQUIRE(buffer[i] == vx_array_get_u8(array, i));
    }

    buffer = {};

    for (size_t i = 0; i < 20; ++i) {
        REQUIRE(vx_array_get_u8(array, i) == (i == 3 ? 8 : 1));
    }

    vx_array_free(array);
}

TEST_CASE("Struct array creation", "[array]") {
    vx_error *error = nullptr;

    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;

    const vx_array *field_array = vx_array_new_null(5);
    CHECK(field_array != nullptr);
    vx_struct_column_builder *builder = vx_struct_column_builder_new(&validity, 2);
    CHECK(builder != nullptr);

    vx_struct_column_builder_add_field(builder, "age", field_array, &error);
    vx_array_free(field_array);

    SECTION("Struct array builder free") {
        vx_struct_column_builder_free(builder);
    }

    SECTION("Struct array builder finalize") {
        const vx_array *struct_array = vx_struct_column_builder_finalize(builder, &error);
        vx_array_free(struct_array);
    }
}
