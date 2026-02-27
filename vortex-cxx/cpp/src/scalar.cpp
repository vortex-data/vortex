// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/scalar.hpp"

#include "vortex_cxx_bridge/lib.h"

namespace vortex::scalar {

Scalar Bool(bool value) {
    return Scalar(ffi::bool_scalar_new(value));
}

Scalar Int8(int8_t value) {
    return Scalar(ffi::i8_scalar_new(value));
}

Scalar Int16(int16_t value) {
    return Scalar(ffi::i16_scalar_new(value));
}

Scalar Int32(int32_t value) {
    return Scalar(ffi::i32_scalar_new(value));
}

Scalar Int64(int64_t value) {
    return Scalar(ffi::i64_scalar_new(value));
}

Scalar Uint8(uint8_t value) {
    return Scalar(ffi::u8_scalar_new(value));
}

Scalar Uint16(uint16_t value) {
    return Scalar(ffi::u16_scalar_new(value));
}

Scalar Uint32(uint32_t value) {
    return Scalar(ffi::u32_scalar_new(value));
}

Scalar Uint64(uint64_t value) {
    return Scalar(ffi::u64_scalar_new(value));
}

Scalar Float32(float value) {
    return Scalar(ffi::f32_scalar_new(value));
}

Scalar Float64(double value) {
    return Scalar(ffi::f64_scalar_new(value));
}

Scalar String(std::string_view value) {
    return Scalar(ffi::string_scalar_new(rust::Str(value.data(), value.length())));
}

Scalar Binary(const uint8_t *data, size_t length) {
    return Scalar(ffi::binary_scalar_new(rust::Slice<const uint8_t>(data, length)));
}

Scalar Cast(Scalar scalar, dtype::DType dtype) {
    return Scalar(std::move(scalar).IntoImpl()->cast_scalar(*std::move(dtype).GetImpl()));
}

} // namespace vortex::scalar