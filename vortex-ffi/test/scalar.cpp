// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vortex.h>
#include "common.h"

using namespace std::string_view_literals;

TEST_CASE("Primitive getters", "[scalar]") {
    vx_scalar *u64 = vx_scalar_new_u64(UINT64_MAX, true);
    defer {
        vx_scalar_free(u64);
    };
    REQUIRE(vx_scalar_get_u64(u64) == UINT64_MAX);

    vx_scalar *f64 = vx_scalar_new_f64(1.5, false);
    defer {
        vx_scalar_free(f64);
    };
    REQUIRE(vx_scalar_get_f64(f64) == 1.5);
}

TEST_CASE("String getters", "[scalar]") {
    vx_error *error = nullptr;

    const std::string_view text = "hello"sv;
    vx_scalar *utf8 = vx_scalar_new_utf8(vx_view {text.data(), text.size()}, false, &error);
    require_no_error(error);
    defer {
        vx_scalar_free(utf8);
    };
    REQUIRE(to_string_view(vx_scalar_get_utf8(utf8)) == text);

    const uint8_t bytes[] = {0xde, 0xad, 0xbe, 0xef};
    vx_scalar *binary = vx_scalar_new_binary(bytes, sizeof(bytes), false, &error);
    require_no_error(error);
    defer {
        vx_scalar_free(binary);
    };
    const vx_view view = vx_scalar_get_binary(binary);
    REQUIRE(view.len == sizeof(bytes));
    REQUIRE(std::memcmp(view.ptr, bytes, sizeof(bytes)) == 0);
}

TEST_CASE("Decimal getters", "[scalar]") {
    vx_error *error = nullptr;

    vx_scalar *d32 = vx_scalar_new_decimal_i32(1234, 5, 2, false, &error);
    require_no_error(error);
    defer {
        vx_scalar_free(d32);
    };
    REQUIRE(vx_scalar_get_decimal_i32(d32) == 1234);

    vx_scalar *d64 = vx_scalar_new_decimal_i64(99999, 12, 3, false, &error);
    require_no_error(error);
    defer {
        vx_scalar_free(d64);
    };
    REQUIRE(vx_scalar_get_decimal_i64(d64) == 99999);
}
