// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/common.hpp"
#include "vortex/error.hpp"
#include "vortex/scalar.hpp"

#include <vortex.h>

#if __STDCPP_FLOAT16_T__ == 1
#include <bit>
#endif
#include <string_view>

namespace vortex {

using detail::Access;
using detail::throw_on_error;

void Scalar::Deleter::operator()(vx_scalar *ptr) const noexcept {
    vx_scalar_free(ptr);
}

Scalar::Scalar(const Scalar &other) : handle_(vx_scalar_clone(other.handle_.get())) {
}

Scalar &Scalar::operator=(const Scalar &other) {
    if (this != &other) {
        handle_.reset(vx_scalar_clone(other.handle_.get()));
    }
    return *this;
}

bool Scalar::is_null() const {
    return vx_scalar_is_null(handle_.get());
}

DataType Scalar::dtype() const {
    return Access::adopt<DataType>(vx_scalar_dtype(handle_.get()));
}

namespace detail {

Scalar adopt(vx_scalar *raw) {
    return Access::adopt<Scalar>(raw);
}

vx_scalar *make_bool(bool value, bool nullable) {
    return vx_scalar_new_bool(value, nullable);
}

vx_scalar *make_primitive(vx_ptype ptype, const void *value, bool nullable) {
    switch (ptype) {
    case PTYPE_U8:
        return vx_scalar_new_u8(*static_cast<const uint8_t *>(value), nullable);
    case PTYPE_U16:
        return vx_scalar_new_u16(*static_cast<const uint16_t *>(value), nullable);
    case PTYPE_U32:
        return vx_scalar_new_u32(*static_cast<const uint32_t *>(value), nullable);
    case PTYPE_U64:
        return vx_scalar_new_u64(*static_cast<const uint64_t *>(value), nullable);
    case PTYPE_I8:
        return vx_scalar_new_i8(*static_cast<const int8_t *>(value), nullable);
    case PTYPE_I16:
        return vx_scalar_new_i16(*static_cast<const int16_t *>(value), nullable);
    case PTYPE_I32:
        return vx_scalar_new_i32(*static_cast<const int32_t *>(value), nullable);
    case PTYPE_I64:
        return vx_scalar_new_i64(*static_cast<const int64_t *>(value), nullable);
#if __STDCPP_FLOAT16_T__ != 1
    case PTYPE_F16:
        return vx_scalar_new_f16_bits(static_cast<const float16_t *>(value)->bits, nullable);
#else
    case PTYPE_F16:
        return vx_scalar_new_f16_bits(std::bit_cast<uint16_t>(*static_cast<const float16_t *>(value)),
                                      nullable);
#endif
    case PTYPE_F32:
        return vx_scalar_new_f32(*static_cast<const float *>(value), nullable);
    case PTYPE_F64:
        return vx_scalar_new_f64(*static_cast<const double *>(value), nullable);
    }
    throw VortexException("unsupported ptype", ErrorCode::InvalidArgument);
}

vx_scalar *make_utf8(std::string_view value, bool nullable) {
    vx_error *error = nullptr;
    vx_scalar *out = vx_scalar_new_utf8(to_view(value), nullable, &error);
    throw_on_error(error);
    return out;
}

vx_scalar *make_binary(BinaryView value, bool nullable) {
    vx_error *error = nullptr;
    vx_scalar *out =
        vx_scalar_new_binary(reinterpret_cast<const uint8_t *>(value.data()), value.size(), nullable, &error);
    throw_on_error(error);
    return out;
}

} // namespace detail

namespace scalar {
Scalar null(const DataType &dtype) {
    vx_error *error = nullptr;
    vx_scalar *out = vx_scalar_new_null(Access::c_ptr(dtype), &error);
    throw_on_error(error);
    return Access::adopt<Scalar>(out);
}
} // namespace scalar
} // namespace vortex
