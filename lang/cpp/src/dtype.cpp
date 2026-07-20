// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"

#include <vortex.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vortex {

using detail::Access;
using detail::throw_on_error;
using detail::to_view;
using namespace std::string_literals;

void DataType::Deleter::operator()(const vx_dtype *ptr) const noexcept {
    vx_dtype_free(ptr);
}

DataType::DataType(const vx_dtype *owned) : handle_(owned) {
}

DataType::DataType(const DataType &other) : handle_(vx_dtype_clone(other.handle_.get())) {
}

DataType &DataType::operator=(const DataType &other) {
    if (this != &other) {
        handle_.reset(vx_dtype_clone(other.handle_.get()));
    }
    return *this;
}

DataType DataType::from_arrow(ArrowSchema *schema) {
    vx_error *error = nullptr;
    const vx_dtype *dtype = vx_dtype_from_arrow_schema(schema, &error);
    throw_on_error(error);
    return DataType(dtype);
}

ArrowSchema DataType::to_arrow() const {
    ArrowSchema schema {};
    vx_error *error = nullptr;
    vx_dtype_to_arrow_schema(handle_.get(), &schema, &error);
    throw_on_error(error);
    return schema;
}

DataTypeVariant DataType::variant() const {
    return static_cast<DataTypeVariant>(vx_dtype_get_variant(handle_.get()));
}

bool DataType::nullable() const {
    return vx_dtype_is_nullable(handle_.get());
}

PType DataType::primitive_type() const {
    return static_cast<PType>(vx_dtype_primitive_ptype(handle_.get()));
}

uint8_t DataType::decimal_precision() const {
    return vx_dtype_decimal_precision(handle_.get());
}

int8_t DataType::decimal_scale() const {
    return vx_dtype_decimal_scale(handle_.get());
}

namespace {
const vx_struct_fields *struct_fields_or_throw(const vx_dtype *dtype) {
    const vx_struct_fields *fields = vx_dtype_struct_dtype(dtype);
    if (fields == nullptr) {
        throw VortexException("dtype is not a struct", ErrorCode::MismatchedTypes);
    }
    return fields;
}
} // namespace

std::vector<StructField> DataType::fields() const {
    const std::unique_ptr<const vx_struct_fields, decltype(&vx_struct_fields_free)> fields(
        struct_fields_or_throw(handle_.get()),
        &vx_struct_fields_free);
    const uint64_t fields_size = vx_struct_fields_nfields(fields.get());
    std::vector<StructField> out;
    out.reserve(fields_size);
    for (uint64_t idx = 0; idx < fields_size; ++idx) {
        const vx_view name = vx_struct_fields_field_name(fields.get(), idx);
        if (name.ptr == nullptr) {
            throw VortexException("error getting field name at index "s + std::to_string(idx),
                                  ErrorCode::Other);
        }
        const vx_dtype *dtype = vx_struct_fields_field_dtype(fields.get(), idx);
        if (dtype == nullptr) {
            throw VortexException("error getting dtype at index "s + std::to_string(idx), ErrorCode::Other);
        }
        out.push_back(StructField {{name.ptr, name.len}, DataType(dtype)});
    }
    return out;
}

DataType DataType::list_element() const {
    const vx_dtype *element = vx_dtype_list_element(handle_.get());
    if (element == nullptr) {
        throw VortexException("dtype is not a list", ErrorCode::MismatchedTypes);
    }
    return DataType(element);
}

DataType DataType::fixed_size_list_element() const {
    const vx_dtype *element = vx_dtype_fixed_size_list_element(handle_.get());
    if (element == nullptr) {
        throw VortexException("dtype is not a fixed-size list", ErrorCode::MismatchedTypes);
    }
    return DataType(element);
}

uint32_t DataType::fixed_size_list_size() const {
    return vx_dtype_fixed_size_list_size(handle_.get());
}

namespace dtype {

DataType null() {
    return Access::adopt<DataType>(vx_dtype_new_null());
}
DataType boolean(bool nullable) {
    return Access::adopt<DataType>(vx_dtype_new_bool(nullable));
}
DataType primitive(PType ptype, bool nullable) {
    return Access::adopt<DataType>(vx_dtype_new_primitive(static_cast<vx_ptype>(ptype), nullable));
}
DataType int8(bool nullable) {
    return primitive(PType::I8, nullable);
}
DataType int16(bool nullable) {
    return primitive(PType::I16, nullable);
}
DataType int32(bool nullable) {
    return primitive(PType::I32, nullable);
}
DataType int64(bool nullable) {
    return primitive(PType::I64, nullable);
}
DataType uint8(bool nullable) {
    return primitive(PType::U8, nullable);
}
DataType uint16(bool nullable) {
    return primitive(PType::U16, nullable);
}
DataType uint32(bool nullable) {
    return primitive(PType::U32, nullable);
}
DataType uint64(bool nullable) {
    return primitive(PType::U64, nullable);
}
DataType float16(bool nullable) {
    return primitive(PType::F16, nullable);
}
DataType float32(bool nullable) {
    return primitive(PType::F32, nullable);
}
DataType float64(bool nullable) {
    return primitive(PType::F64, nullable);
}
DataType utf8(bool nullable) {
    return Access::adopt<DataType>(vx_dtype_new_utf8(nullable));
}
DataType binary(bool nullable) {
    return Access::adopt<DataType>(vx_dtype_new_binary(nullable));
}
DataType decimal(uint8_t precision, int8_t scale, bool nullable) {
    return Access::adopt<DataType>(vx_dtype_new_decimal(precision, scale, nullable));
}
DataType list(DataType element, bool nullable) {
    return Access::adopt<DataType>(vx_dtype_new_list(Access::release(std::move(element)), nullable));
}
DataType fixed_size_list(DataType element, uint32_t size, bool nullable) {
    return Access::adopt<DataType>(
        vx_dtype_new_fixed_size_list(Access::release(std::move(element)), size, nullable));
}

DataType struct_(std::span<const StructField> fields, bool nullable) {
    vx_error *error = nullptr;
    std::unique_ptr<vx_struct_fields_builder, decltype(&vx_struct_fields_builder_free)> handle(
        vx_struct_fields_builder_new(),
        vx_struct_fields_builder_free);

    for (const auto &[name, dtype] : fields) {
        vx_struct_fields_builder_add_field(handle.get(),
                                           to_view(name),
                                           vx_dtype_clone(Access::c_ptr(dtype)),
                                           &error);
        throw_on_error(error);
    }
    vx_struct_fields *ffi_fields = vx_struct_fields_builder_finalize(handle.release());
    return Access::adopt<DataType>(vx_dtype_new_struct(ffi_fields, nullable));
}

DataType struct_(std::initializer_list<StructField> fields, bool nullable) {
    return struct_({fields.begin(), fields.end()}, nullable);
}

} // namespace dtype
} // namespace vortex
