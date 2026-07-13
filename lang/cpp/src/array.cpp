// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/array.hpp"
#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"

#include <vortex.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace vortex {

using detail::Access;
using detail::throw_on_error;

Validity::Validity(const Validity &other)
    : type_(other.type_), array_(other.array_ != nullptr ? vx_array_clone(other.array_) : nullptr) {
}

Validity::Validity(ValidityType type) : type_(type), array_(nullptr) {
    if (type == ValidityType::Array) {
        throw VortexException("Validity(ValidityType) called with ValidityType::Array",
                              ErrorCode::InvalidArgument);
    }
}

Validity::Validity(ValidityType type, const Array &array)
    : type_(type), array_(vx_array_clone(Access::c_ptr<Array>(array))) {
}

Validity::Validity(Validity &&other) noexcept : type_(other.type_), array_(other.array_) {
    other.array_ = nullptr;
    other.type_ = ValidityType::NonNullable;
}

Validity &Validity::operator=(const Validity &other) {
    if (this != &other) {
        *this = Validity(other);
    }
    return *this;
}

Validity &Validity::operator=(Validity &&other) noexcept {
    if (this != &other) {
        vx_array_free(array_);
        type_ = other.type_;
        array_ = other.array_;
        other.array_ = nullptr;
        other.type_ = ValidityType::NonNullable;
    }
    return *this;
}

Validity::~Validity() {
    vx_array_free(array_);
}

Array Validity::array() const {
    if (type_ != ValidityType::Array || array_ == nullptr) {
        throw VortexException("validity has no backing array", ErrorCode::InvalidArgument);
    }
    return Access::adopt<Array>(vx_array_clone(array_));
}

Validity ValidityArray(const Array &bools) {
    return Validity(ValidityType::Array, vx_array_clone(Access::c_ptr(bools)));
}

bool detail::ValidityBits::is_null(size_t index) const noexcept {
    if (all_invalid_) {
        return true;
    }
    if (bits_ == nullptr) {
        return false;
    }
    const size_t bit = bit_offset_ + index;
    return (bits_[bit / 8] >> (bit % 8) & 1) == 0;
}

namespace detail {
ValidityBits::ValidityBits(const Session &session, const vx_array *canonical) {
    vx_validity raw {};
    vx_error *error = nullptr;
    vx_array_get_validity(canonical, &raw, &error);
    throw_on_error(error);

    switch (static_cast<ValidityType>(raw.type)) {
    case ValidityType::NonNullable:
    case ValidityType::AllValid:
        return;
    case ValidityType::AllInvalid:
        all_invalid_ = true;
        return;
    case ValidityType::Array:
        break;
    }

    owner_ = vx_array_canonicalize(Access::c_ptr(session), raw.array, &error);
    vx_array_free(raw.array);
    throw_on_error(error);

    bits_ = static_cast<const uint8_t *>(vx_array_data_ptr_bool(owner_, &bit_offset_, &error));
    if (error != nullptr) {
        vx_array_free(owner_);
    }
    throw_on_error(error);
}

ValidityBits::ValidityBits(ValidityBits &&other) noexcept
    : owner_(other.owner_), bits_(other.bits_), bit_offset_(other.bit_offset_),
      all_invalid_(other.all_invalid_) {
    other.owner_ = nullptr;
    other.bits_ = nullptr;
}

ValidityBits &ValidityBits::operator=(ValidityBits &&other) noexcept {
    if (this != &other) {
        vx_array_free(owner_);
        owner_ = other.owner_;
        bits_ = other.bits_;
        bit_offset_ = other.bit_offset_;
        all_invalid_ = other.all_invalid_;
        other.owner_ = nullptr;
        other.bits_ = nullptr;
    }
    return *this;
}

ValidityBits::~ValidityBits() {
    vx_array_free(owner_);
}
} // namespace detail

static const vx_struct_fields *struct_fields_or_throw(const vx_dtype *dtype) {
    const vx_struct_fields *fields = vx_dtype_struct_dtype(dtype);
    if (fields == nullptr) {
        throw VortexException("dtype is not a struct", ErrorCode::MismatchedTypes);
    }
    return fields;
}

void Array::Deleter::operator()(const vx_array *ptr) const noexcept {
    vx_array_free(ptr);
}

Array::Array(const Array &other) : handle_(vx_array_clone(other.handle_.get())) {
}

Array &Array::operator=(const Array &other) {
    if (this != &other) {
        handle_.reset(vx_array_clone(other.handle_.get()));
    }
    return *this;
}

Array Array::null(size_t len) {
    return Access::adopt<Array>(vx_array_new_null(len));
}

Array Array::primitive_raw(vx_ptype ptype, const void *data, size_t len, const Validity &validity) {
    std::optional<Array> keep_alive;
    vx_validity raw {};
    raw.type = static_cast<vx_validity_type>(validity.type());
    if (validity.type() == ValidityType::Array) {
        keep_alive = validity.array();
        raw.array = Access::c_ptr(*keep_alive);
    }

    vx_error *error = nullptr;
    const vx_array *out = vx_array_new_primitive(ptype, data, len, &raw, &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

Array Array::from_arrow(ArrowArray *array, ArrowSchema *schema, bool nullable) {
    vx_error *error = nullptr;
    const vx_array *out = vx_array_from_arrow(array, schema, nullable, &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

size_t Array::size() const {
    return vx_array_len(handle_.get());
}

bool Array::nullable() const {
    return vx_array_is_nullable(handle_.get());
}

bool Array::has_dtype(DataTypeVariant v) const {
    return vx_array_has_dtype(handle_.get(), static_cast<vx_dtype_variant>(v));
}

bool Array::is_primitive(PType p) const {
    return vx_array_is_primitive(handle_.get(), static_cast<vx_ptype>(p));
}

DataType Array::dtype() const {
    return Access::adopt<DataType>(vx_array_dtype(handle_.get()));
}

Validity Array::validity() const {
    vx_validity raw {};
    vx_error *error = nullptr;
    vx_array_get_validity(handle_.get(), &raw, &error);
    throw_on_error(error);
    return Access::adopt<Validity>(static_cast<ValidityType>(raw.type), raw.array);
}

size_t Array::null_count() const {
    vx_error *error = nullptr;
    const size_t count = vx_array_invalid_count(handle_.get(), &error);
    throw_on_error(error);
    return count;
}

Array Array::field(size_t index) const {
    vx_error *error = nullptr;
    const vx_array *out = vx_array_get_field(handle_.get(), index, &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

Array Array::field(std::string_view name) const {
    const DataType dt = dtype();
    const std::unique_ptr<const vx_struct_fields, decltype(&vx_struct_fields_free)> fields(
        struct_fields_or_throw(detail::Access::c_ptr(dt)),
        &vx_struct_fields_free);
    const uint64_t fields_size = vx_struct_fields_nfields(fields.get());
    for (uint64_t i = 0; i < fields_size; ++i) {
        const vx_view field = vx_struct_fields_field_name(fields.get(), i);
        if (std::string_view {field.ptr, field.len} == name) {
            return this->field(i);
        }
    }
    throw VortexException("no field named \"" + std::string(name) + "\"", ErrorCode::InvalidArgument);
}

Array Array::slice(size_t begin, size_t end) const {
    vx_error *error = nullptr;
    const vx_array *out = vx_array_slice(handle_.get(), begin, end, &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

Array Array::apply(const Expression &expr) const {
    vx_error *error = nullptr;
    const vx_array *out = vx_array_apply(handle_.get(), Access::c_ptr(expr), &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

Array Array::canonicalize(const Session &session) const {
    vx_error *error = nullptr;
    const vx_array *out = vx_array_canonicalize(Access::c_ptr(session), handle_.get(), &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

Array make_struct(std::initializer_list<std::pair<std::string_view, Array>> fields,
                  const Validity &validity) {
    StructArrayBuilder builder(validity, fields.size());
    for (const auto &[name, field] : fields) {
        builder.add(name, field);
    }
    return std::move(builder).build();
}

void StructArrayBuilder::Deleter::operator()(vx_struct_column_builder *ptr) const noexcept {
    vx_struct_column_builder_free(ptr);
}

StructArrayBuilder::StructArrayBuilder(const Validity &validity, size_t capacity) {
    std::optional<Array> keep_alive;
    vx_validity raw {};
    raw.type = static_cast<vx_validity_type>(validity.type());
    if (validity.type() == ValidityType::Array) {
        keep_alive = validity.array();
        raw.array = Access::c_ptr(*keep_alive);
    }
    handle_.reset(vx_struct_column_builder_new(&raw, capacity));
}

StructArrayBuilder &StructArrayBuilder::add(std::string_view name, const Array &field) & {
    vx_error *error = nullptr;
    vx_struct_column_builder_add_field(handle_.get(), detail::to_view(name), Access::c_ptr(field), &error);
    throw_on_error(error);
    return *this;
}

StructArrayBuilder &&StructArrayBuilder::add(std::string_view name, const Array &field) && {
    add(name, field);
    return std::move(*this);
}

Array StructArrayBuilder::build() && {
    vx_error *error = nullptr;
    const vx_array *out = vx_struct_column_builder_finalize(handle_.release(), &error);
    throw_on_error(error);
    return Access::adopt<Array>(out);
}

PrimitiveView<bool> Array::bools(const Session &session) const {
    Array canonical = canonicalize(session);
    const vx_array *raw = Access::c_ptr(canonical);
    if (!vx_array_has_dtype(raw, DTYPE_BOOL)) {
        throw VortexException("bools(): array is not a Bool array", ErrorCode::MismatchedTypes);
    }
    detail::ValidityBits validity(session, raw);
    const size_t len = vx_array_len(raw);
    return PrimitiveView<bool>(std::move(canonical), std::move(validity), len);
}

StringView Array::strings(const Session &session) const {
    Array canonical = canonicalize(session);
    const vx_array *raw = Access::c_ptr(canonical);
    if (!vx_array_has_dtype(raw, DTYPE_UTF8)) {
        throw VortexException("strings(): array is not a Utf8 array", ErrorCode::MismatchedTypes);
    }
    detail::ValidityBits validity(session, raw);
    const size_t len = vx_array_len(raw);
    return StringView(std::move(canonical), std::move(validity), len);
}

BytesView Array::bytes(const Session &session) const {
    Array canonical = canonicalize(session);
    const vx_array *raw = Access::c_ptr(canonical);
    if (!vx_array_has_dtype(raw, DTYPE_BINARY)) {
        throw VortexException("bytes(): array is not a Binary array", ErrorCode::MismatchedTypes);
    }
    detail::ValidityBits validity(session, raw);
    const size_t len = vx_array_len(raw);
    return BytesView(std::move(canonical), std::move(validity), len);
}

bool PrimitiveView<bool>::value(size_t i) const {
    return vx_array_get_bool(Access::c_ptr(canonical_), i);
}

std::string_view StringView::operator[](size_t i) const {
    vx_error *error = nullptr;
    const vx_view out = vx_array_utf8_at(Access::c_ptr(canonical_), i, &error);
    throw_on_error(error);
    return {out.ptr, out.len};
}

BinaryView BytesView::operator[](size_t i) const {
    vx_error *error = nullptr;
    const vx_view out = vx_array_binary_at(Access::c_ptr(canonical_), i, &error);
    throw_on_error(error);
    return {reinterpret_cast<const std::byte *>(out.ptr), out.len};
}
} // namespace vortex
