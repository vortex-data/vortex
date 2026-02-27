// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <string_view>
#include "dtype.hpp"
#include "vortex_cxx_bridge/lib.h"

namespace vortex::scalar {
class Scalar {
public:
    Scalar() = delete;
    explicit Scalar(rust::Box<ffi::Scalar> impl) : impl(std::move(impl)) {
    }
    Scalar(Scalar &&other) noexcept = default;
    Scalar &operator=(Scalar &&other) noexcept = default;
    ~Scalar() = default;

    Scalar(const Scalar &) = delete;
    Scalar &operator=(const Scalar &) = delete;

    rust::Box<ffi::Scalar> IntoImpl() && {
        return std::move(impl);
    }

private:
    rust::Box<ffi::Scalar> impl;
};

// Factory functions for creating scalar values
Scalar Bool(bool value);
Scalar Int8(int8_t value);
Scalar Int16(int16_t value);
Scalar Int32(int32_t value);
Scalar Int64(int64_t value);
Scalar Uint8(uint8_t value);
Scalar Uint16(uint16_t value);
Scalar Uint32(uint32_t value);
Scalar Uint64(uint64_t value);
Scalar Float32(float value);
Scalar Float64(double value);
Scalar String(std::string_view value);
Scalar Binary(const uint8_t *data, size_t length);
/// TODO: Other Scalars are only supported by casting for now.
Scalar Cast(Scalar scalar, dtype::DType dtype);
} // namespace vortex::scalar
