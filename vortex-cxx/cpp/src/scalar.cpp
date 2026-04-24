// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/scalar.hpp"

#include "vortex_cxx_bridge/lib.h"

namespace vortex::scalar {

Scalar bool_(bool value) {
    return Scalar(ffi::bool_scalar_new(value));
}

Scalar int8(int8_t value) {
    return Scalar(ffi::i8_scalar_new(value));
}

Scalar int16(int16_t value) {
    return Scalar(ffi::i16_scalar_new(value));
}

Scalar int32(int32_t value) {
    return Scalar(ffi::i32_scalar_new(value));
}

Scalar int64(int64_t value) {
    return Scalar(ffi::i64_scalar_new(value));
}

Scalar uint8(uint8_t value) {
    return Scalar(ffi::u8_scalar_new(value));
}

Scalar uint16(uint16_t value) {
    return Scalar(ffi::u16_scalar_new(value));
}

Scalar uint32(uint32_t value) {
    return Scalar(ffi::u32_scalar_new(value));
}

Scalar uint64(uint64_t value) {
    return Scalar(ffi::u64_scalar_new(value));
}

Scalar float32(float value) {
    return Scalar(ffi::f32_scalar_new(value));
}

Scalar float64(double value) {
    return Scalar(ffi::f64_scalar_new(value));
}

Scalar string(std::string_view value) {
    return Scalar(ffi::string_scalar_new(rust::Str(value.data(), value.length())));
}

Scalar binary(const uint8_t *data, size_t length) {
    return Scalar(ffi::binary_scalar_new(rust::Slice<const uint8_t>(data, length)));
}

Scalar cast(Scalar scalar, dtype::DType dtype) {
    return Scalar(std::move(scalar).IntoImpl()->cast_scalar(*std::move(dtype).GetImpl()));
}

} // namespace vortex::scalar