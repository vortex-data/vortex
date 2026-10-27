// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <vortex.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace vortex {
enum class ErrorCode {
    Other = VX_ERROR_CODE_OTHER,
    Panic = VX_ERROR_CODE_PANIC,
};

class VortexException : public std::runtime_error {
public:
    explicit VortexException(const std::string &message, ErrorCode code = ErrorCode::Other)
        : std::runtime_error(message), code_(code) {
    }

    ErrorCode code() const {
        return code_;
    }

private:
    ErrorCode code_;
};

namespace detail {
// Throw VortexException and free "error" if it is non-nullptr.
inline void throw_on_error(vx_error *error) {
    if (error == nullptr) {
        return;
    }
    const vx_view str = vx_error_message(error);
    const std::string message {str.ptr, str.len};
    const auto code = static_cast<ErrorCode>(vx_error_get_code(error));
    vx_error_free(error);
    throw VortexException(message, code);
}

inline vx_view to_view(std::string_view view) {
    return {view.data(), view.size()};
}
} // namespace detail
} // namespace vortex
