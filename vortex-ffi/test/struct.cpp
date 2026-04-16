// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <vortex.h>

using namespace std::string_view_literals;
using namespace std::string_literals;

TEST_CASE("Struct builder", "[struct]") {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();

    constexpr auto col1 = "col1"sv;
    const vx_string *col1_name = vx_string_new(col1.data(), col1.size());
    const vx_dtype *col1_dtype = vx_dtype_new_primitive(PTYPE_U8, false);
    vx_struct_fields_builder_add_field(builder, col1_name, col1_dtype);

    constexpr auto col2 = "col2"sv;
    const vx_string *col2_name = vx_string_new(col2.data(), col2.size());
    const vx_dtype *col2_dtype = vx_dtype_new_binary(true);
    vx_struct_fields_builder_add_field(builder, col2_name, col2_dtype);

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

    for (size_t i = 0; i < STRUCT_LEN; ++i) {
        const std::string target_name = "name"s + std::to_string(i);
        const vx_string *name = vx_string_new(target_name.data(), target_name.size());
        const vx_dtype *dtype = i % 2 ? vx_dtype_new_binary(false) : vx_dtype_new_primitive(PTYPE_F32, true);
        vx_struct_fields_builder_add_field(builder, name, dtype);
    }
    vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    REQUIRE(fields != nullptr);

    const size_t len = vx_struct_fields_nfields(fields);
    CHECK(len == STRUCT_LEN);
    for (size_t i = 0; i < len; ++i) {
        // borrowed
        const vx_string *name = vx_struct_fields_field_name(fields, i);
        // owned TODO(myrrc): that's weird API
        const vx_dtype *dtype = vx_struct_fields_field_dtype(fields, i);

        std::string_view name_view {vx_string_ptr(name), vx_string_len(name)};
        std::string target_name = "name"s + std::to_string(i);

        CHECK(name_view == target_name);

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
