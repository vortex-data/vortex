// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <vortex.h>

#include "common.h"

using namespace std::string_view_literals;
using namespace std::string_literals;

TEST_CASE("Struct builder", "[struct]") {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();
    vx_error *error = nullptr;

    const vx_dtype *col1_dtype = vx_dtype_new_primitive(PTYPE_U8, false);
    vx_struct_fields_builder_add_field(builder, vx_view_from_cstr("col1"), col1_dtype, &error);
    require_no_error(error);

    const vx_dtype *col2_dtype = vx_dtype_new_binary(true);
    vx_struct_fields_builder_add_field(builder, vx_view_from_cstr("col2"), col2_dtype, &error);
    require_no_error(error);

    SECTION("Struct builder free") {
        vx_struct_fields_builder_free(builder);
    }

    SECTION("Struct builder finalize") {
        vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);

        SECTION("struct fields free") {
            vx_struct_fields_free(fields);
        }

        SECTION("struct fields finalize") {
            const vx_dtype *dtype = vx_dtype_new_struct(fields, false);
            vx_dtype_free(dtype);
        }
    }
}

constexpr size_t STRUCT_LEN = 10;
TEST_CASE("Creating structs", "[struct]") {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();
    REQUIRE(builder != nullptr);
    vx_error *error = nullptr;

    for (size_t i = 0; i < STRUCT_LEN; ++i) {
        const std::string target_name = "name"s + std::to_string(i);
        const vx_dtype *dtype = i % 2 ? vx_dtype_new_binary(false) : vx_dtype_new_primitive(PTYPE_F32, true);
        vx_struct_fields_builder_add_field(builder, vx_view_from_cstr(target_name.c_str()), dtype, &error);
        require_no_error(error);
    }
    vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    REQUIRE(fields != nullptr);

    const size_t len = vx_struct_fields_nfields(fields);
    CHECK(len == STRUCT_LEN);
    for (size_t i = 0; i < len; ++i) {
        const vx_view name = vx_struct_fields_field_name(fields, i);
        const vx_dtype *dtype = vx_struct_fields_field_dtype(fields, i);

        std::string target_name = "name"s + std::to_string(i);
        CHECK(to_string_view(name) == target_name);

        if (i % 2) {
            CHECK_FALSE(vx_dtype_is_nullable(dtype));
            CHECK(vx_dtype_get_variant(dtype) == DTYPE_BINARY);
        } else {
            CHECK(vx_dtype_is_nullable(dtype));
            CHECK(vx_dtype_get_variant(dtype) == DTYPE_PRIMITIVE);
        }

        vx_dtype_free(dtype);
    }

    vx_struct_fields_free(fields);
}
