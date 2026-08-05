// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"

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

    /**
     * Read scalar's value.
     *
     * Supported types are bool, primitives, std::string_view, and BinaryView.
     *
     * std::string_view and BinaryView returned borrow from scalar and stay
     * valid while scalar is valid.
     *
     * Throws if scalar type does not match T or value is Null.
     */
    template <element_type T>
    T get() const;

    /**
     * Read a decimal scalar's unscaled value.
     * int8/16/32/64_t are supported.
     * Throws if scalar is not a decimal, is Null, or value does not fit in T.
     */
    template <primitive_type T>
    T get_decimal() const;

private:
    friend struct detail::Access;
    explicit Scalar(const vx_scalar *owned) : handle_(owned) {
    }
    const vx_scalar *release() && {
        return handle_.release();
    }

    struct Deleter {
        void operator()(const vx_scalar *ptr) const noexcept;
    };
    std::unique_ptr<const vx_scalar, Deleter> handle_;
};

template <element_type T>
T Scalar::get() const {
    using enum DataTypeVariant;
    using enum ErrorCode;
    using enum PType;

    const DataType data_type = dtype();
    const DataTypeVariant variant = data_type.variant();
    if (variant == Null) {
        throw VortexException("DataType is null", InvalidArgument);
    }

    const vx_scalar *const h = handle_.get();
    if constexpr (std::is_same_v<T, bool>) {
        if (variant != Bool) {
            throw VortexException("scalar get type doesn't match", InvalidArgument);
        }
        return vx_scalar_get_bool(h);
    } else if constexpr (std::is_same_v<T, std::string_view>) {
        if (variant != Utf8) {
            throw VortexException("scalar get type doesn't match", InvalidArgument);
        }
        vx_view v = vx_scalar_get_utf8(h);
        return std::string_view(v.ptr, v.len);
    } else if constexpr (std::is_same_v<T, BinaryView>) {
        if (variant != Binary) {
            throw VortexException("scalar get type doesn't match", InvalidArgument);
        }
        vx_view v = vx_scalar_get_binary(h);
        return BinaryView(reinterpret_cast<const std::byte *>(v.ptr), v.len);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        if (variant != Primitive || data_type.primitive_type() != U8) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_u8(h);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        if (variant != Primitive || data_type.primitive_type() != U16) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_u16(h);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        if (variant != Primitive || data_type.primitive_type() != U32) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_u32(h);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        if (variant != Primitive || data_type.primitive_type() != U64) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_u64(h);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        if (variant != Primitive || data_type.primitive_type() != I8) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_i8(h);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        if (variant != Primitive || data_type.primitive_type() != I16) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_i16(h);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (variant != Primitive || data_type.primitive_type() != I32) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_i32(h);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        if (variant != Primitive || data_type.primitive_type() != I64) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_i64(h);
    } else if constexpr (std::is_same_v<T, float>) {
        if (variant != Primitive || data_type.primitive_type() != F32) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_f32(h);
    } else if constexpr (std::is_same_v<T, double>) {
        if (variant != Primitive || data_type.primitive_type() != F64) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        return vx_scalar_get_f64(h);
    } else if constexpr (std::is_same_v<T, float16_t>) {
        if (variant != Primitive || data_type.primitive_type() != F16) {
            throw VortexException("scalar get type doesn't match", ErrorCode::InvalidArgument);
        }
        const uint16_t bits = vx_scalar_get_f16_bits(h);
#if __STDCPP_FLOAT16_T__ != 1
        return {.bits = bits};
#else
        return std::bit_cast<float16_t>(bits);
#endif
    } else {
        static_assert(false, "scalar get is not supported for this type");
    }
}

template <primitive_type T>
T Scalar::get_decimal() const {
    if (dtype().variant() != DataTypeVariant::Decimal) {
        throw VortexException("DataType is not Decimal", ErrorCode::InvalidArgument);
    }
    const vx_scalar *const h = handle_.get();
    if constexpr (std::is_same_v<T, int8_t>) {
        return vx_scalar_get_decimal_i8(h);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return vx_scalar_get_decimal_i16(h);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return vx_scalar_get_decimal_i32(h);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return vx_scalar_get_decimal_i64(h);
    } else {
        static_assert(false, "unsupported decimal scalar get type");
    }
}

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

template <primitive_type T>
Scalar decimal(T value, uint8_t precision, int8_t scale, bool nullable = false) {
    vx_error *error = nullptr;
    vx_scalar *out = nullptr;
    if constexpr (std::is_same_v<T, int8_t>) {
        out = vx_scalar_new_decimal_i8(value, precision, scale, nullable, &error);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        out = vx_scalar_new_decimal_i16(value, precision, scale, nullable, &error);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        out = vx_scalar_new_decimal_i32(value, precision, scale, nullable, &error);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        out = vx_scalar_new_decimal_i64(value, precision, scale, nullable, &error);
    } else {
        static_assert(false, "can't construct decimal of following scale");
    }
    detail::throw_on_error(error);
    return detail::Access::adopt<Scalar>(out);
}

// A typed null of (a nullable copy of) a given DataType.
Scalar null(const DataType &dtype);
} // namespace scalar
} // namespace vortex
