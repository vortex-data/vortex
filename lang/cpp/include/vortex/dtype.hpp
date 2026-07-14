// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"

#include <vortex.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vortex {

struct StructField;

enum class DataTypeVariant {
    Null = DTYPE_NULL,
    Bool = DTYPE_BOOL,
    // Primitives e.g., u8, i16, f32
    Primitive = DTYPE_PRIMITIVE,
    // Variable-length UTF-8 string
    Utf8 = DTYPE_UTF8,
    // Variable-length binary
    Binary = DTYPE_BINARY,
    // Nested struct
    Struct = DTYPE_STRUCT,
    // Nested list
    List = DTYPE_LIST,
    // User-defined extension
    Extension = DTYPE_EXTENSION,
    // Decimal with fixed precision and scale
    Decimal = DTYPE_DECIMAL,
    // Nested fixed-size list
    FixedSizeList = DTYPE_FIXED_SIZE_LIST,
};

// Primitive type
enum class PType {
    U8 = PTYPE_U8,
    U16 = PTYPE_U16,
    U32 = PTYPE_U32,
    U64 = PTYPE_U64,
    I8 = PTYPE_I8,
    I16 = PTYPE_I16,
    I32 = PTYPE_I32,
    I64 = PTYPE_I64,
    F16 = PTYPE_F16,
    F32 = PTYPE_F32,
    F64 = PTYPE_F64,
};

/**
 * A Vortex data type. Data types are logical: they say nothing about physical
 * representation.
 */
class DataType {
public:
    DataType(const DataType &other);
    DataType(DataType &&) noexcept = default;
    DataType &operator=(const DataType &other);
    DataType &operator=(DataType &&) noexcept = default;

    /**
     * Consume an ArrowSchema and convert it into a DataType. The schema must
     * not be used after this call.
     */
    static DataType from_arrow(ArrowSchema *schema);

    /**
     * Convert dtype to an Arrow C schema. Caller is responsible for invoking
     * schema's release() callback.
     */
    ArrowSchema to_arrow() const;

    DataTypeVariant variant() const;
    bool nullable() const;

    PType primitive_type() const;
    uint8_t decimal_precision() const;
    int8_t decimal_scale() const;

    /**
     * For a Struct dtype, return its fields in order.
     * Throws if DataType is not Struct.
     */
    std::vector<StructField> fields() const;

    // List accessors. Valid only on List and FixedSizeList dtypes

    DataType list_element() const;
    DataType fixed_size_list_element() const;
    uint32_t fixed_size_list_size() const;

private:
    friend struct detail::Access;
    explicit DataType(const vx_dtype *owned);
    const vx_dtype *release() && {
        return handle_.release();
    }

    struct Deleter {
        void operator()(const vx_dtype *ptr) const noexcept;
    };
    std::unique_ptr<const vx_dtype, Deleter> handle_;
};

// Field of a Struct DataType.
struct StructField {
    std::string name;
    DataType dtype;
};

namespace dtype {

inline constexpr bool Nullable = true;

DataType null();
DataType boolean(bool nullable = false);
DataType primitive(PType ptype, bool nullable = false);
DataType int8(bool nullable = false);
DataType int16(bool nullable = false);
DataType int32(bool nullable = false);
DataType int64(bool nullable = false);
DataType uint8(bool nullable = false);
DataType uint16(bool nullable = false);
DataType uint32(bool nullable = false);
DataType uint64(bool nullable = false);
DataType float16(bool nullable = false);
DataType float32(bool nullable = false);
DataType float64(bool nullable = false);
DataType utf8(bool nullable = false);
DataType binary(bool nullable = false);
DataType decimal(uint8_t precision, int8_t scale, bool nullable = false);
DataType list(DataType element, bool nullable = false);
DataType fixed_size_list(DataType element, uint32_t size, bool nullable = false);

/**
 * Create a DataTypeVariant::Struct from a field list.
 *
 * Example:
 *
 * using dtype::Nullable;
 * DataType dtype = dtype::struct_({
 *     {"age", dtype::uint8()},
 *     {"height", dtype::uint16(Nullable)}}
 * );
 */
DataType struct_(std::span<const StructField> fields, bool nullable = false);
DataType struct_(std::initializer_list<StructField> fields, bool nullable = false);
} // namespace dtype
} // namespace vortex
