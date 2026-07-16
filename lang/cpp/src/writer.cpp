// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/array.hpp"
#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"
#include "vortex/writer.hpp"
#include "vortex/session.hpp"

#include <initializer_list>
#include <vortex.h>

#include <string_view>

namespace vortex {

using detail::Access;
using detail::throw_on_error;
using detail::to_view;

void Writer::Deleter::operator()(vx_array_sink *ptr) const noexcept {
    vx_array_sink_abort(ptr);
}

Writer Writer::open(const Session &session, std::string_view path, const DataType &dtype) {
    vx_error *error = nullptr;
    vx_array_sink *sink =
        vx_array_sink_open_file(Access::c_ptr(session), to_view(path), Access::c_ptr(dtype), &error);
    throw_on_error(error);
    return Writer(sink);
}

void Writer::push(std::span<const Array> arrays) {
    if (handle_ == nullptr) {
        throw VortexException("null handle_", ErrorCode::InvalidArgument);
    }
    vx_error *error = nullptr;
    for (const Array &array : arrays) {
        vx_array_sink_push(handle_.get(), Access::c_ptr(array), &error);
        throw_on_error(error);
    }
}

void Writer::push(const Array &array) {
    push(std::span {&array, 1});
}

void Writer::push(std::initializer_list<Array> arrays) {
    push({arrays.begin(), arrays.end()});
}

void Writer::finish() {
    if (handle_ == nullptr) {
        throw VortexException("finish() called twice", ErrorCode::InvalidArgument);
    }
    vx_error *error = nullptr;
    vx_array_sink_close(handle_.release(), &error);
    throw_on_error(error);
}
} // namespace vortex
