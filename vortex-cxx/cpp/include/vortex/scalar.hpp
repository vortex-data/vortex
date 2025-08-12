// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <string_view>
#include "dtype.hpp"
#include "vortex_cxx_bridge/lib.h"

namespace vortex {

class Scalar {
public:
    Scalar() = delete;
    Scalar(Scalar &&other) noexcept;
    Scalar &operator=(Scalar &&other) noexcept;
    ~Scalar();

    Scalar(const Scalar &) = delete;
    Scalar &operator=(const Scalar &) = delete;

    // Factory functions for creating scalar values
    static Scalar bool_(bool value);
    static Scalar int8(int8_t value);
    static Scalar int16(int16_t value);
    static Scalar int32(int32_t value);
    static Scalar int64(int64_t value);
    static Scalar uint8(uint8_t value);
    static Scalar uint16(uint16_t value);
    static Scalar uint32(uint32_t value);
    static Scalar uint64(uint64_t value);
    static Scalar float32(float value);
    static Scalar float64(double value);
    static Scalar string(std::string_view value);
    static Scalar binary(const uint8_t *data, size_t length);
    static Scalar cast(Scalar scalar, DType dtype);

private:
    friend class Expr;
    explicit Scalar(rust::Box<ffi::Scalar> impl);
    rust::Box<ffi::Scalar> impl_;
};

} // namespace vortex