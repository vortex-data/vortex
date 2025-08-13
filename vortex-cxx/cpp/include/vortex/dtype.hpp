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

class DType {
public:
    DType() = delete;
    DType(DType &&other) noexcept : impl_(std::move(other.impl_)) {
    }
    DType &operator=(DType &&other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
    ~DType() = default;

    DType(const DType &) = delete;
    DType &operator=(const DType &) = delete;

    // Factory functions
    static DType null();
    static DType bool_(bool nullable = false);
    static DType primitive(PType ptype, bool nullable = false);
    static DType int8(bool nullable = false);
    static DType int16(bool nullable = false);
    static DType int32(bool nullable = false);
    static DType int64(bool nullable = false);
    static DType uint8(bool nullable = false);
    static DType uint16(bool nullable = false);
    static DType uint32(bool nullable = false);
    static DType uint64(bool nullable = false);
    static DType float16(bool nullable = false);
    static DType float32(bool nullable = false);
    static DType float64(bool nullable = false);
    static DType decimal(uint8_t precision = 10, int8_t scale = 0, bool nullable = false);
    static DType utf8(bool nullable = false);
    static DType binary(bool nullable = false);
    static DType from_arrow(struct ArrowSchema &schema, bool non_nullable = false);

    // Methods
    std::string to_string() const;

private:
    friend class Scalar;
    explicit DType(rust::Box<ffi::DType> impl) : impl_(std::move(impl)) {
    }
    rust::Box<ffi::DType> impl_;
};

} // namespace vortex