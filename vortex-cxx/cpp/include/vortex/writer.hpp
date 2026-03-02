// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <nanoarrow/common/inline_types.h>
#include "vortex_cxx_bridge/lib.h"

namespace vortex {

class VortexWriteOptions;

class VortexWriter {
public:
    VortexWriter(VortexWriter &&other) noexcept = default;
    VortexWriter &operator=(VortexWriter &&other) noexcept = default;
    ~VortexWriter() = default;

    VortexWriter(const VortexWriter &) = delete;
    VortexWriter &operator=(const VortexWriter &) = delete;

    void PushArrayStream(ArrowArrayStream &stream);

    uint64_t BytesWritten() const;

    uint64_t BufferedBytes() const;

    void Finish();

private:
    friend class VortexWriteOptions;

    explicit VortexWriter(rust::Box<ffi::VortexWriter> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexWriter> impl_;
};

} // namespace vortex
