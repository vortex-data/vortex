// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"
#include "vortex/expression.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace vortex {

// Types that a PrimitiveView can hold
template <class T>
concept primitive_view = primitive_type<T> || std::is_same_v<T, bool>;

template <primitive_view T>
class PrimitiveView;

class Array;
class StringView;
class BytesView;

/*
 * Validity type tells us whether there are null/invalid values in an Array.
 */
enum class ValidityType {
    // Items can't be null
    NonNullable = VX_VALIDITY_NON_NULLABLE,
    // All items are valid
    AllValid = VX_VALIDITY_ALL_VALID,
    // All items are invalid
    AllInvalid = VX_VALIDITY_ALL_INVALID,
    // Item validity is set in a boolean array: true = valid, false = invalid
    Array = VX_VALIDITY_ARRAY,
};

/**
 * Array per-element validity of type ValidityType.
 * If ValidityType is ValidityType::Array, holds a boolean array
 * with validity items.
 *
 * You can use shortcut constants NonNullable/AllValid/AllInvalid and
 * function ValidityArray(validity_bools);
 */
class Validity {
public:
    // NonNullable/AllValid/AllInvalid constructor
    // NOLINTNEXTLINE(google-explicit-constructor)
    Validity(ValidityType type);
    // Validity determined by a boolean array, true = valid, false = invalid.
    static Validity from_array(const Array &bools);

    Validity(const Validity &other);
    Validity(Validity &&other) noexcept;
    Validity &operator=(const Validity &other);
    Validity &operator=(Validity &&other) noexcept;
    ~Validity();

    ValidityType type() const {
        return type_;
    }

    // Boolean validity array. Throws if type() != ValidityType::Array.
    Array array() const;

private:
    friend struct detail::Access;
    friend Validity ValidityArray(const Array &bools);
    Validity(ValidityType type, const vx_array *owned) : type_(type), array_(owned) {
    }

    ValidityType type_;
    const vx_array *array_;
};

namespace detail {
// Validity bitmap for typed views. Owns the arrays that back the bits
class ValidityBits {
public:
    ValidityBits(ValidityBits &&other) noexcept;
    ValidityBits &operator=(ValidityBits &&other) noexcept;
    ValidityBits(const ValidityBits &) = delete;
    ValidityBits &operator=(const ValidityBits &) = delete;
    ~ValidityBits();

    bool is_null(size_t index) const;

private:
    friend class vortex::Array;
    // Materialize validity of "canonical"
    ValidityBits(const Session &session, const vx_array *canonical);

    const vx_array *owner_ = nullptr;
    const uint8_t *bits_ = nullptr;
    size_t bit_offset_ = 0;
    bool all_invalid_ = false;
};
} // namespace detail

// A reference-counted handle to columnar data in some encoding
class Array {
public:
    Array(const Array &other);
    Array(Array &&) noexcept = default;
    Array &operator=(const Array &other);
    Array &operator=(Array &&) noexcept = default;

    // An all-null array with DataType Null.
    static Array null(size_t len);

    /**
     * A Primitive array copied from a typed buffer.
     *
     * Example:
     *
     * std::array<uint16_t, 3> buffer = {0, 1, 2};
     * auto array = Array::primitive(buffer);
     */
    template <primitive_type T>
    static Array primitive(std::span<const T> data, const Validity &validity = ValidityType::NonNullable) {
        return primitive_raw(detail::to_ptype<T>(), data.data(), data.size(), validity);
    }

    /**
     * Import an Arrow array. Consumes both "array" and "schema", do not use
     * or release them afterwards. For a record batch pass nullable = false.
     */
    static Array from_arrow(ArrowArray *array, ArrowSchema *schema, bool nullable);

    size_t size() const;
    bool nullable() const;
    bool has_dtype(DataTypeVariant variant) const;
    bool is_primitive(PType ptype) const;
    DataType dtype() const;
    Validity validity() const;

    // Number of null/invalid elements in Array
    size_t null_count() const;

    /**
     * Get a Struct field by index. Throws if Array is not a Struct or if index
     * is out of bounds.
     */
    Array field(size_t index) const;

    /**
     * Get a Struct field by name. Throws if Array is not a Struct or doesn't
     * have this named field.
     */
    Array field(std::string_view name) const;

    /*
     * Create a new Array slicing [begin; end) rows from original.
     * Doesn't copy the original buffer or sliced buffer.
     *
     * Example:
     *
     * std::array<uint16_t, 3> buffer = {0, 1, 2};
     * Array array = Array::primitive<uint16_t>(buffer);
     * Array sliced = array.slice(1, 2);
     */
    Array slice(size_t begin, size_t end) const;

    /**
     * Apply an expression to an array.
     *
     * This function operates in constant time and doesn't execute the result
     * array. To execute the array, canonicalise it.
     *
     * Example:
     *
     * using namespace vortex::expr::ops;
     *
     * std::array<uint16_t, 3> buffer = {0, 1, 2};
     * Array array = Array::primitive<uint16_t>(buffer);
     * Expression expr = expr::root() > expr::lit<uint16_t>(0);
     * Array result = array.apply(expr);
     */
    Array apply(const Expression &expr) const;

    /**
     * Bulk view over values. Canonicalizes the array.
     * Throws if T does not match Array's ptype.
     *
     * Example:
     *
     * Session session;
     * std::array<uint16_t, 3> buffer = {0, 1, 2};
     * Array array = Array::primitive(buffer);
     * auto view = array.values(session);
     */
    template <primitive_type T>
    PrimitiveView<T> values(const Session &session) const;

    // Bulk view over Bool values
    PrimitiveView<bool> bools(const Session &session) const;

    // Bulk view over Utf8 values.
    StringView strings(const Session &session) const;

    // Bulk view over Binary values.
    BytesView bytes(const Session &session) const;

private:
    friend struct detail::Access;
    friend class StringView;
    friend class BytesView;
    template <primitive_view T>
    friend class PrimitiveView;

    explicit Array(const vx_array *owned) : handle_(owned) {
    }
    const vx_array *release() && {
        return handle_.release();
    }

    static Array primitive_raw(vx_ptype ptype, const void *data, size_t len, const Validity &validity);
    Array canonicalize(const Session &session) const;

    struct Deleter {
        void operator()(const vx_array *ptr) const noexcept;
    };
    std::unique_ptr<const vx_array, Deleter> handle_;
};

/**
 * Create a Struct array from named columns of equal length.
 *
 * Example:
 *
 * using enum ValidityType;
 * std::array<uint16_t, 3> age_buffer = {0, 1, 2};
 * std::array<uint32_t, 3> height_buffer = {0, 1, 2};
 * Array ages = Array::primitive(age_buffer);
 * Array heights = Array::primitive(height_buffer);
 * Array result = make_struct(
 *     {{"age", ages}, {"height", heights}},
 *     NonNullable);
 */
Array make_struct(std::initializer_list<std::pair<std::string_view, Array>> fields,
                  const Validity &validity = ValidityType::NonNullable);

/**
 * Create a Struct Array from a dynamic number of fields.
 * Prefer make_struct() if you know all fields beforehand.
 *
 * Example:
 *
 * std::array<uint16_t, 3> age_buffer = {0, 1, 2};
 * std::array<uint32_t, 3> height_buffer = {0, 1, 2};
 * Array ages = Array::primitive(age_buffer);
 * Array heights = Array::primitive(height_buffer);
 *
 * StructArrayBuilder b(NonNullable, 2);
 * Array result = builder
 *     .add("age", ages)
 *     .add("height", heights)
 *     .build();
 */
class StructArrayBuilder {
public:
    explicit StructArrayBuilder(const Validity &validity, size_t capacity = 0);
    StructArrayBuilder(const StructArrayBuilder &) = delete;
    StructArrayBuilder &operator=(const StructArrayBuilder &) = delete;
    StructArrayBuilder(StructArrayBuilder &&) noexcept = default;
    StructArrayBuilder &operator=(StructArrayBuilder &&) noexcept = default;

    StructArrayBuilder &add(std::string_view name, const Array &field) &;
    StructArrayBuilder &&add(std::string_view name, const Array &field) &&;

    // Consume the builder and return and Array
    Array build() &&;

private:
    struct Deleter {
        void operator()(vx_struct_column_builder *ptr) const noexcept;
    };
    std::unique_ptr<vx_struct_column_builder, Deleter> handle_;
};

/**
 * Typed read-only view over a Primitive array.
 *
 * Owns a canonicalized copy of Array. values() and anything derived from
 * it are valid as long as the view lives.
 */
template <primitive_view T>
class PrimitiveView {
public:
    /*
     * Get raw values from this view. Values at null/invalid positions are
     * unspecified.
     */
    std::span<const T> values() const {
        return {data_, size_};
    }
    bool is_null(size_t index) const {
        return validity_.is_null(index);
    }
    size_t size() const {
        return size_;
    }

private:
    friend class Array;
    PrimitiveView(Array canonical, detail::ValidityBits validity, const T *data, size_t size)
        : canonical_(std::move(canonical)), validity_(std::move(validity)), data_(data), size_(size) {
    }

    Array canonical_;
    detail::ValidityBits validity_;
    const T *data_;
    size_t size_;
};

/**
 * Read-only view over a Bool array. As Bool values are bit-packed, there's no
 * span. Read individual values with value(i).
 */
template <>
class PrimitiveView<bool> {
public:
    /*
     * Get raw value from this view. Values at null/invalid positions are
     * unspecified.
     */
    bool value(size_t index) const;
    bool is_null(size_t index) const {
        return validity_.is_null(index);
    }
    size_t size() const {
        return size_;
    }

private:
    friend class Array;
    PrimitiveView(Array canonical, detail::ValidityBits validity, size_t size)
        : canonical_(std::move(canonical)), validity_(std::move(validity)), size_(size) {
    }

    Array canonical_;
    detail::ValidityBits validity_;
    size_t size_;
};

/**
 * Read-only view over a Utf8 array.
 *
 * operator[] is O(1) and borrows from the view's canonical copy. Returned
 * string_views are valid as long as the view lives.
 */
class StringView {
public:
    /*
     * Get raw value from this view. Values at null/invalid positions are
     * unspecified.
     */
    std::string_view operator[](size_t index) const;
    bool is_null(size_t index) const {
        return validity_.is_null(index);
    }
    size_t size() const {
        return size_;
    }

private:
    friend class Array;
    StringView(Array canonical, detail::ValidityBits validity, size_t size)
        : canonical_(std::move(canonical)), validity_(std::move(validity)), size_(size) {
    }

    Array canonical_;
    detail::ValidityBits validity_;
    size_t size_;
};

/**
 * Read-only view over a Bytes array.
 *
 * Byte spans borrow from the view's canonical copy and are valid as long as
 * the view lives.
 */
class BytesView {
public:
    /*
     * Get raw value from this view. Values at null/invalid positions are
     * unspecified.
     */
    BinaryView operator[](size_t index) const;
    bool is_null(size_t index) const {
        return validity_.is_null(index);
    }
    size_t size() const {
        return size_;
    }

private:
    friend class Array;
    BytesView(Array canonical, detail::ValidityBits validity, size_t size)
        : canonical_(std::move(canonical)), validity_(std::move(validity)), size_(size) {
    }

    Array canonical_;
    detail::ValidityBits validity_;
    size_t size_;
};

template <primitive_type T>
PrimitiveView<T> Array::values(const Session &session) const {
    Array canonical = canonicalize(session);
    const vx_array *raw = detail::Access::c_ptr(canonical);
    if (!vx_array_is_primitive(raw, detail::to_ptype<T>())) {
        throw VortexException("values<T>: T does not match the array's ptype", ErrorCode::MismatchedTypes);
    }
    vx_error *error = nullptr;
    const void *data = vx_array_data_ptr_primitive(raw, &error);
    detail::throw_on_error(error);
    detail::ValidityBits validity(session, raw);
    const size_t n = vx_array_len(raw);
    return PrimitiveView<T>(std::move(canonical), std::move(validity), static_cast<const T *>(data), n);
}
} // namespace vortex
