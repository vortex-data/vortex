// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/scalar.hpp"

#include "vortex_cxx_bridge/lib.h"

namespace vortex {

// Factory functions
Scalar Scalar::bool_(bool value) {
    return Scalar(ffi::bool_scalar_new(value));
}

Scalar Scalar::int8(int8_t value) {
    return Scalar(ffi::i8_scalar_new(value));
}

Scalar Scalar::int16(int16_t value) {
    return Scalar(ffi::i16_scalar_new(value));
}

Scalar Scalar::int32(int32_t value) {
    return Scalar(ffi::i32_scalar_new(value));
}

Scalar Scalar::int64(int64_t value) {
    return Scalar(ffi::i64_scalar_new(value));
}

Scalar Scalar::uint8(uint8_t value) {
    return Scalar(ffi::u8_scalar_new(value));
}

Scalar Scalar::uint16(uint16_t value) {
    return Scalar(ffi::u16_scalar_new(value));
}

Scalar Scalar::uint32(uint32_t value) {
    return Scalar(ffi::u32_scalar_new(value));
}

Scalar Scalar::uint64(uint64_t value) {
    return Scalar(ffi::u64_scalar_new(value));
}

Scalar Scalar::float32(float value) {
    return Scalar(ffi::f32_scalar_new(value));
}

Scalar Scalar::float64(double value) {
    return Scalar(ffi::f64_scalar_new(value));
}

Scalar Scalar::string(std::string_view value) {
    return Scalar(ffi::string_scalar_new(rust::Str(value.data(), value.length())));
}

Scalar Scalar::binary(const uint8_t *data, size_t length) {
    return Scalar(ffi::binary_scalar_new(rust::Slice<const uint8_t>(data, length)));
}

Scalar Scalar::cast(Scalar scalar, DType dtype) {
    return Scalar(scalar.impl_->cast_scalar(*dtype.impl_));
}

} // namespace vortex