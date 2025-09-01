// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "rust/cxx.h"

namespace vortex {

VortexFile VortexFile::Open(const std::string &path) {
    try {
        return VortexFile(ffi::open_file(path));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

uint64_t VortexFile::RowCount() const {
    return impl_->row_count();
}

ScanBuilder VortexFile::CreateScanBuilder() const {
    return ScanBuilder(impl_->scan_builder());
}

} // namespace vortex