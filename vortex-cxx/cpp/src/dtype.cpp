// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/dtype.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"

namespace vortex {

// Factory functions
DType DType::null() {
    return DType(ffi::dtype_null());
}

DType DType::bool_(bool nullable) {
    return DType(ffi::dtype_bool(nullable));
}

DType DType::primitive(PType ptype, bool nullable) {
    return DType(ffi::dtype_primitive(static_cast<ffi::PType>(ptype), nullable));
}

DType DType::int8(bool nullable) {
    return primitive(PType::I8, nullable);
}

DType DType::int16(bool nullable) {
    return primitive(PType::I16, nullable);
}

DType DType::int32(bool nullable) {
    return primitive(PType::I32, nullable);
}

DType DType::int64(bool nullable) {
    return primitive(PType::I64, nullable);
}

DType DType::uint8(bool nullable) {
    return primitive(PType::U8, nullable);
}

DType DType::uint16(bool nullable) {
    return primitive(PType::U16, nullable);
}

DType DType::uint32(bool nullable) {
    return primitive(PType::U32, nullable);
}

DType DType::uint64(bool nullable) {
    return primitive(PType::U64, nullable);
}

DType DType::float16(bool nullable) {
    return primitive(PType::F16, nullable);
}

DType DType::float32(bool nullable) {
    return primitive(PType::F32, nullable);
}

DType DType::float64(bool nullable) {
    return primitive(PType::F64, nullable);
}

DType DType::decimal(uint8_t precision, int8_t scale, bool nullable) {
    return DType(ffi::dtype_decimal(precision, scale, nullable));
}

DType DType::utf8(bool nullable) {
    return DType(ffi::dtype_utf8(nullable));
}

DType DType::binary(bool nullable) {
    return DType(ffi::dtype_binary(nullable));
}

DType DType::from_arrow(struct ArrowSchema &schema, bool non_nullable) {
    try {
        return DType(ffi::from_arrow(reinterpret_cast<uint8_t *>(&schema), non_nullable));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

// Methods
std::string DType::to_string() const {
    auto rust_str = impl_->to_string();
    return std::string(rust_str.data(), rust_str.length());
}

} // namespace vortex