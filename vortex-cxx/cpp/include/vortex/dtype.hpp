// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <string>
#include <nanoarrow/common/inline_types.h>
#include "vortex_cxx_bridge/lib.h"

namespace vortex {

enum class PType : uint8_t {
    U8 = 0,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
};

namespace dtype {
    class DType {
    public:
        DType() = delete;
        explicit DType(rust::Box<ffi::DType> impl) : impl(std::move(impl)) {
        }
        DType(DType &&other) noexcept = default;
        DType &operator=(DType &&other) = default;
        ~DType() = default;

        DType(const DType &) = delete;
        DType &operator=(const DType &) = delete;

        std::string ToString() const;

        const rust::Box<ffi::DType> &GetImpl() {
            return impl;
        }

    private:
        rust::Box<ffi::DType> impl;
    };

    // Factory functions
    DType Null();
    DType Bool(bool nullable = false);
    DType Primitive(PType ptype, bool nullable = false);
    DType Int8(bool nullable = false);
    DType Int16(bool nullable = false);
    DType Int32(bool nullable = false);
    DType Int64(bool nullable = false);
    DType Uint8(bool nullable = false);
    DType Uint16(bool nullable = false);
    DType Uint32(bool nullable = false);
    DType Uint64(bool nullable = false);
    DType Float16(bool nullable = false);
    DType Float32(bool nullable = false);
    DType Float64(bool nullable = false);
    DType Decimal(uint8_t precision = 10, int8_t scale = 0, bool nullable = false);
    DType Utf8(bool nullable = false);
    DType Binary(bool nullable = false);
    /// TODO: Other DTypes are only supported by creating from Arrow for now.
    DType FromArrow(struct ArrowSchema &schema, bool non_nullable = false);
} // namespace dtype

} // namespace vortex