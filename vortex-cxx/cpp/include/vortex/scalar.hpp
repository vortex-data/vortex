// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <string_view>
#include "dtype.hpp"
#include "vortex_cxx_bridge/lib.h"

namespace vortex::scalar {
using dtype::DType;
class Scalar {
public:
    Scalar() = delete;
    explicit Scalar(rust::Box<ffi::Scalar> impl) : impl_(std::move(impl)) {
    }
    Scalar(Scalar &&other) noexcept = default;
    Scalar &operator=(Scalar &&other) noexcept = default;
    ~Scalar() = default;

    Scalar(const Scalar &) = delete;
    Scalar &operator=(const Scalar &) = delete;

    rust::Box<ffi::Scalar> IntoImpl() && {
        return std::move(impl_);
    }

private:
    rust::Box<ffi::Scalar> impl_;
};

// Factory functions for creating scalar values
Scalar bool_(bool value);
Scalar int8(int8_t value);
Scalar int16(int16_t value);
Scalar int32(int32_t value);
Scalar int64(int64_t value);
Scalar uint8(uint8_t value);
Scalar uint16(uint16_t value);
Scalar uint32(uint32_t value);
Scalar uint64(uint64_t value);
Scalar float32(float value);
Scalar float64(double value);
Scalar string(std::string_view value);
Scalar binary(const uint8_t *data, size_t length);
/// TODO: Other Scalars are only supported by casting for now.
Scalar cast(Scalar scalar, DType dtype);
} // namespace vortex::scalar
