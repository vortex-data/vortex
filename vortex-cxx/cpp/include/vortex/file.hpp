// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <nanoarrow/common/inline_types.h>
#include "vortex_cxx_bridge/lib.h"

namespace vortex {
class ScanBuilder;

class VortexFile {
public:
    static VortexFile Open(const std::string &path);

    VortexFile(VortexFile &&other) noexcept = default;
    VortexFile &operator=(VortexFile &&other) noexcept = default;
    ~VortexFile() = default;

    VortexFile(const VortexFile &) = delete;
    VortexFile &operator=(const VortexFile &) = delete;

    /// Get the number of rows in the file.
    uint64_t RowCount() const;

    /// Create a scan builder for the file.
    /// The scan builder can be used to scan the file.
    ScanBuilder CreateScanBuilder() const;

private:
    explicit VortexFile(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexFile> impl_;
};

} // namespace vortex