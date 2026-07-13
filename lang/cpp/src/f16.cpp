// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/common.hpp"

#include <cstdint>
#include <cstring>

namespace vortex {

#if __STDCPP_FLOAT16_T__ != 1
float16_t::operator float() const {
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
    std::memcpy(&result, &out, sizeof(result));
    return result;
}
#endif
} // namespace vortex
