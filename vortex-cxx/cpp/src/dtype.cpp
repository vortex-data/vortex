// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/dtype.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"

namespace vortex::dtype {
DType Null() {
    return DType(ffi::dtype_null());
}

DType Bool(bool nullable) {
    return DType(ffi::dtype_bool(nullable));
}

DType Primitive(PType ptype, bool nullable) {
    return DType(ffi::dtype_primitive(static_cast<ffi::PType>(ptype), nullable));
}

DType Int8(bool nullable) {
    return Primitive(PType::I8, nullable);
}

DType Int16(bool nullable) {
    return Primitive(PType::I16, nullable);
}

DType Int32(bool nullable) {
    return Primitive(PType::I32, nullable);
}

DType Int64(bool nullable) {
    return Primitive(PType::I64, nullable);
}

DType Uint8(bool nullable) {
    return Primitive(PType::U8, nullable);
}

DType Uint16(bool nullable) {
    return Primitive(PType::U16, nullable);
}

DType Uint32(bool nullable) {
    return Primitive(PType::U32, nullable);
}

DType Uint64(bool nullable) {
    return Primitive(PType::U64, nullable);
}

DType Float16(bool nullable) {
    return Primitive(PType::F16, nullable);
}

DType Float32(bool nullable) {
    return Primitive(PType::F32, nullable);
}

DType Float64(bool nullable) {
    return Primitive(PType::F64, nullable);
}

DType Decimal(uint8_t precision, int8_t scale, bool nullable) {
    return DType(ffi::dtype_decimal(precision, scale, nullable));
}

DType Utf8(bool nullable) {
    return DType(ffi::dtype_utf8(nullable));
}

DType Binary(bool nullable) {
    return DType(ffi::dtype_binary(nullable));
}

DType FromArrow(struct ArrowSchema &schema, bool non_nullable) {
    try {
        return DType(ffi::from_arrow(reinterpret_cast<uint8_t *>(&schema), non_nullable));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}
// Methods
std::string DType::ToString() const {
    auto rust_str = impl->to_string();
    return std::string(rust_str.data(), rust_str.length());
}
} // namespace vortex::dtype
