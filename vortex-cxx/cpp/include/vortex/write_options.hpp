// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <nanoarrow/common/inline_types.h>
#include "vortex_cxx_bridge/lib.h"

namespace vortex {

class VortexWriteOptions {
public:
    VortexWriteOptions() : impl_(ffi::write_options_new()) {
    }
    VortexWriteOptions(VortexWriteOptions &&other) noexcept : impl_(std::move(other.impl_)) {
    }
    VortexWriteOptions &operator=(VortexWriteOptions &&other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
    ~VortexWriteOptions() = default;

    VortexWriteOptions(const VortexWriteOptions &) = delete;
    VortexWriteOptions &operator=(const VortexWriteOptions &) = delete;

    /// Write an ArrowArrayStream to a Vortex file
    void WriteArrayStream(ArrowArrayStream &stream, const std::string &path);

private:
    rust::Box<ffi::VortexWriteOptions> impl_;
};

} // namespace vortex