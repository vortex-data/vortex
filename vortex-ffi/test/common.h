// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <catch2/catch_test_macros.hpp>
#include <string>
#include "vortex.h"

inline std::string to_string(vx_error *err) {
    const vx_view msg = vx_error_message(err);
    return {msg.ptr, msg.len};
}

inline std::string_view to_string_view(vx_view str) {
    return {str.ptr, str.len};
}

inline void require_no_error(vx_error *error, bool assert = true) {
    if (!error) {
        return;
    }
    std::string message = to_string(error);
    vx_error_free(error);
    if (assert) {
        FAIL(message);
    } else {
        throw std::runtime_error(message);
    }
}

inline uint8_t array_get_u8(vx_session *session, const vx_array *array, size_t index) {
    vx_error *error = nullptr;
    const vx_scalar *scalar = vx_array_get_scalar(session, array, index, &error);
    require_no_error(error);
    const uint8_t value = vx_scalar_get_u8(scalar);
    vx_scalar_free(scalar);
    return value;
}

inline uint16_t array_get_u16(vx_session *session, const vx_array *array, size_t index) {
    vx_error *error = nullptr;
    const vx_scalar *scalar = vx_array_get_scalar(session, array, index, &error);
    require_no_error(error);
    const uint16_t value = vx_scalar_get_u16(scalar);
    vx_scalar_free(scalar);
    return value;
}

template <class F>
struct Defer {
    Defer(F &&f) : f(std::move(f)) {
    }
    ~Defer() {
        f();
    }
    F f;
};
#define CONCAT(x, y)  x##y
#define CONCAT2(x, y) CONCAT(x, y)
#define defer         Defer CONCAT2(defer_, __LINE__) = [&]
