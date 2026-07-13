// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <unistd.h>
#include "vortex.h"

using namespace std::string_literals;
using namespace std::string_view_literals;

TEST_CASE("Session creation", "[session]") {
    vx_session *session = vx_session_new();
    REQUIRE(session != nullptr);
    vx_session *session2 = vx_session_clone(session);
    REQUIRE(session2 != nullptr);
    REQUIRE(session != session2);
    vx_session_free(session);
    vx_session_free(session2);
}

TEST_CASE("vx_view from C string", "[str]") {
    const std::string_view str = "Широкая строка"sv;
    const std::string owned {str};
    const vx_view view = vx_view_from_cstr(owned.c_str());
    REQUIRE(view.len == str.size());
    REQUIRE(std::string_view {view.ptr, view.len} == str);
}

TEST_CASE("Creating dtypes", "[dtype]") {
    const vx_dtype *dtype = vx_dtype_new_null();
    REQUIRE(dtype != nullptr);
    CHECK(vx_dtype_get_variant(dtype) == DTYPE_NULL);
    CHECK(vx_dtype_is_nullable(dtype));
    vx_dtype_free(dtype);

    dtype = vx_dtype_new_decimal(5, 2, false);
    REQUIRE(dtype != nullptr);
    CHECK(vx_dtype_get_variant(dtype) == DTYPE_DECIMAL);
    CHECK(vx_dtype_decimal_precision(dtype) == 5);
    CHECK(vx_dtype_decimal_scale(dtype) == 2);
    CHECK_FALSE(vx_dtype_is_nullable(dtype));

    CHECK(vx_dtype_struct_dtype(dtype) == nullptr);
    CHECK(vx_dtype_list_element(dtype) == nullptr);

    vx_dtype_free(dtype);
}
