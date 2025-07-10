// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "rust/cxx.h"
#include "vortex-cxx/src/lib.rs.h"

namespace vortex {

class VortexException : public std::runtime_error {
public:
    explicit VortexException(const std::string &message) : std::runtime_error(message) {
    }
};

class ScanBuilder {
public:
    ScanBuilder(rust::Box<ffi::VortexScanBuilder> impl) : impl_(std::move(impl)) {
    }

    /// Set the filter on the scan builder.
    void set_filter(std::string_view filter) {
        try {
            ffi::scan_builder_set_filter(
                *impl_,
                rust::Slice<const uint8_t>(reinterpret_cast<const uint8_t *>(filter.data()), filter.size()));
        } catch (const rust::cxxbridge1::Error &e) {
            throw VortexException(e.what());
        }
    }

    /// Set the limit on the number of rows to scan.
    void set_limit(uint64_t limit) {
        ffi::scan_builder_set_limit(*impl_, limit);
    }

    /// Consume the scan builder to a stream of record batches.
    /// The scan builder is consumed and cannot be used after this call.
    arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> into_stream();

    /// Consume the scan builder to an Arrow array and schema.
    /// The scan builder is consumed and cannot be used after this call.
    std::pair<ArrowArray, ArrowSchema> into_arrow();

private:
    rust::Box<ffi::VortexScanBuilder> impl_;
};

class VortexFile {
public:
    static VortexFile open(const std::string &path) {
        try {
            return VortexFile(ffi::open_file(path));
        } catch (const rust::cxxbridge1::Error &e) {
            throw VortexException(e.what());
        }
    }

    /// Get the number of rows in the file.
    uint64_t row_count() const {
        return ffi::file_row_count(*impl_);
    }

    /// Create a scan builder for the file.
    /// The scan builder can be used to scan the file.
    ScanBuilder scan_builder() const {
        return ScanBuilder(ffi::file_scan_builder(*impl_));
    }

private:
    explicit VortexFile(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexFile> impl_;
};

} // namespace vortex