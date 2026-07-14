// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vortex.h>

#if __STDCPP_FLOAT16_T__ != 1

namespace vortex {
struct float16_t {
    uint16_t bits;
    friend bool operator==(float16_t, float16_t) = default;
    // NOLINTNEXTLINE
    operator float() const;
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
    std::is_same_v<T, double> || std::is_same_v<T, float16_t>;
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
