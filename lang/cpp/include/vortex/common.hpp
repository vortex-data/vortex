// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vortex.h>

#if __STDCPP_FLOAT16_T__ != 1

namespace vortex {
struct float16_t {
    uint16_t bits;
    constexpr friend bool operator==(float16_t, float16_t) = default;
    // NOLINTNEXTLINE
    constexpr operator float() const {
        float result;
        const uint32_t sign = (bits >> 15) & 1;
        const uint32_t exponent = (bits >> 10) & 0x1F;
        const uint32_t mantissa = bits & 0x3FF;

        uint32_t out;
        if (exponent == 0x1F) {
            out = (sign << 31) | 0x7F800000 | (mantissa << 13);
        } else if (exponent == 0) {
            if (mantissa == 0) {
                out = sign << 31;
            } else {
                uint32_t m = mantissa;
                int e = -1;
                do {
                    m <<= 1;
                    ++e;
                } while ((m & 0x400) == 0);
                out = (sign << 31) | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
            }
        } else {
            out = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
        }
        return std::bit_cast<float>(out);
    }
};
static_assert(sizeof(float16_t) == 2 && std::is_trivially_copyable_v<float16_t>);
} // namespace vortex

#else

#include <stdfloat>

namespace vortex {
using float16_t = ::std::float16_t;
} // namespace vortex

#endif

namespace vortex::detail {
struct Access {
    template <class T, class... Args>
    static T adopt(Args &&...args) {
        return T(std::forward<Args>(args)...);
    }
    template <class T>
    static auto release(T &&t) {
        return std::forward<T>(t).release();
    }
    template <class T>
    static auto c_ptr(const T &t) {
        return t.handle_.get();
    }
};

template <class T>
constexpr vx_ptype to_ptype() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return PTYPE_U8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return PTYPE_U16;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return PTYPE_U32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return PTYPE_U64;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return PTYPE_I8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return PTYPE_I16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return PTYPE_I32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return PTYPE_I64;
#if __STDCPP_FLOAT16_T__ == 1
    } else if constexpr (std::is_same_v<T, _Float16>) {
        return PTYPE_F16;
#endif
    } else if constexpr (std::is_same_v<T, float16_t>) {
        return PTYPE_F16;
    } else if constexpr (std::is_same_v<T, float>) {
        return PTYPE_F32;
    } else {
        static_assert(std::is_same_v<T, double>);
        return PTYPE_F64;
    }
}

template <class T>
inline constexpr bool is_numeric_element =
    std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, uint64_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, float> ||
    std::is_same_v<T, double>
#if __STDCPP_FLOAT16_T__ == 1
    || std::is_same_v<T, _Float16>
#else
    || std::is_same_v<T, float16_t>
#endif
    ;
} // namespace vortex::detail

namespace vortex {

// View over a single Binary byte range.
using BinaryView = std::span<const std::byte>;

// Types that can be stored in a Primitive array
template <class T>
concept primitive_type = detail::is_numeric_element<T>;

// Types constructible as scalars or literals.
template <class T>
concept element_type = primitive_type<T> || std::is_same_v<T, bool> || std::is_same_v<T, std::string_view> ||
                       std::is_same_v<T, BinaryView>;
} // namespace vortex
