// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"
#include "vortex/dtype.hpp"

#include <vortex.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

namespace vortex {

// A single value with an associated DataType
class Scalar {
public:
    Scalar(const Scalar &other);
    Scalar(Scalar &&) noexcept = default;
    Scalar &operator=(const Scalar &other);
    Scalar &operator=(Scalar &&) noexcept = default;

    bool is_null() const;
    DataType dtype() const;

private:
    friend struct detail::Access;
    explicit Scalar(vx_scalar *owned) : handle_(owned) {
    }
    vx_scalar *release() && {
        return handle_.release();
    }

    struct Deleter {
        void operator()(vx_scalar *ptr) const noexcept;
    };
    std::unique_ptr<vx_scalar, Deleter> handle_;
};

namespace detail {
vx_scalar *make_bool(bool value, bool nullable);
vx_scalar *make_primitive(vx_ptype ptype, const void *value, bool nullable);
vx_scalar *make_utf8(std::string_view value, bool nullable);
vx_scalar *make_binary(BinaryView value, bool nullable);
Scalar adopt(vx_scalar *raw);
} // namespace detail

namespace scalar {
/**
 * A scalar of DataType selected by T: bool, primitive, string_view (utf8),
 * or a BinaryView (binary).
 *
 * Bytes are copied for utf8 and binary scalars.
 */
template <element_type T>
Scalar of(T value, bool nullable = false) {
    if constexpr (std::is_same_v<T, bool>) {
        return detail::adopt(detail::make_bool(value, nullable));
    } else if constexpr (std::is_same_v<T, std::string_view>) {
        return detail::adopt(detail::make_utf8(value, nullable));
    } else if constexpr (std::is_same_v<T, BinaryView>) {
        return detail::adopt(detail::make_binary(value, nullable));
    } else {
        return detail::adopt(detail::make_primitive(detail::to_ptype<T>(), &value, nullable));
    }
}

// A typed null of (a nullable copy of) a given DataType.
Scalar null(const DataType &dtype);

Scalar decimal_i32(int32_t value, uint8_t precision, int8_t scale, bool nullable = false);
Scalar decimal_i64(int64_t value, uint8_t precision, int8_t scale, bool nullable = false);
} // namespace scalar
} // namespace vortex
