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

TEST_CASE("Creating and iterating binaries", "[binary]") {
    for (std::string_view str : {"ololo"sv, "Широкая строка"sv, "مرحبا بالعالم"sv}) {
        const vx_binary *binary = vx_binary_new(str.data(), str.size());

        REQUIRE(binary != nullptr);
        const size_t len = vx_binary_len(binary);
        REQUIRE(len == str.size());

        const char *ptr = vx_binary_ptr(binary);
        REQUIRE(std::string_view {ptr, len} == str);

        const vx_binary *binary2 = vx_binary_clone(binary);
        vx_binary_free(binary);

        ptr = vx_binary_ptr(binary2);
        REQUIRE(std::string_view {ptr, len} == str);

        vx_binary_free(binary2);
    }
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
