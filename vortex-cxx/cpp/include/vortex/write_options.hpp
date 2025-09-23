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
    VortexWriteOptions(VortexWriteOptions &&other) noexcept = default;
    VortexWriteOptions &operator=(VortexWriteOptions &&other) noexcept = default;
    ~VortexWriteOptions() = default;

    VortexWriteOptions(const VortexWriteOptions &) = delete;
    VortexWriteOptions &operator=(const VortexWriteOptions &) = delete;

    /// Write an ArrowArrayStream to a Vortex file
    void WriteArrayStream(ArrowArrayStream &stream, const std::string &path);

private:
    rust::Box<ffi::VortexWriteOptions> impl_;
};

} // namespace vortex