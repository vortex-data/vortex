// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <string>

// TODO remove
typedef void FFI_ArrowSchema;
typedef void FFI_ArrowArrayStream;

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

inline void require_no_error(vx_error *err) {
    if (err) {
        FAIL(to_string(err));
    }
}
