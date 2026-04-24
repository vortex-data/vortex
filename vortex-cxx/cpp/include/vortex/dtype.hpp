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
        explicit DType(rust::Box<ffi::DType> impl) : impl_(std::move(impl)) {
        }
        DType(DType &&other) noexcept = default;
        DType &operator=(DType &&other) = default;
        ~DType() = default;

        DType(const DType &) = delete;
        DType &operator=(const DType &) = delete;

        std::string ToString() const;

        const rust::Box<ffi::DType> &GetImpl() {
            return impl_;
        }

    private:
        rust::Box<ffi::DType> impl_;
    };

    // Factory functions
    DType null();
    DType bool_(bool nullable = false);
    DType primitive(PType ptype, bool nullable = false);
    DType int8(bool nullable = false);
    DType int16(bool nullable = false);
    DType int32(bool nullable = false);
    DType int64(bool nullable = false);
    DType uint8(bool nullable = false);
    DType uint16(bool nullable = false);
    DType uint32(bool nullable = false);
    DType uint64(bool nullable = false);
    DType float16(bool nullable = false);
    DType float32(bool nullable = false);
    DType float64(bool nullable = false);
    DType decimal(uint8_t precision = 10, int8_t scale = 0, bool nullable = false);
    DType utf8(bool nullable = false);
    DType binary(bool nullable = false);
    /// TODO: Other DTypes are only supported by creating from Arrow for now.
    DType from_arrow(struct ArrowSchema &schema, bool non_nullable = false);
} // namespace dtype

} // namespace vortex