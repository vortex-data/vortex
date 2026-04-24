// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <catch2/catch_test_macros.hpp>
#include <string>
#include "vortex.h"

inline std::string to_string(vx_error *err) {
    const vx_string *msg = vx_error_get_message(err);
    return {vx_string_ptr(msg), vx_string_len(msg)};
}

inline std::string_view to_string_view(const vx_string *msg) {
    return {vx_string_ptr(msg), vx_string_len(msg)};
}

inline std::string_view to_string_view(vx_error *err) {
    return to_string_view(vx_error_get_message(err));
}

inline void require_no_error(vx_error *error, bool assert = true) {
    if (!error) {
        return;
    }
    auto message = to_string(error);
    vx_error_free(error);
    if (assert) {
        FAIL(message);
    } else {
        throw std::runtime_error(message);
    }
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
