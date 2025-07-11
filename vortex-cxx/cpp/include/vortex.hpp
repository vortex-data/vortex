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

/// TODO(xinyu): better error handling
class VortexException : public std::runtime_error {
public:
    explicit VortexException(const std::string &message) : std::runtime_error(message) {
    }
};

/// TODO(xinyu): This should be a builder/option pattern
/// Configure the tokio runtime with the specified number of worker threads
/// If the runtime has already been initialized, this function will throw an exception.
inline void ConfigureRuntime(size_t worker_threads) {
    try {
        ffi::configure_runtime(worker_threads);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

class ScanBuilder {
public:
    /// Set the filter on the scan builder.
    ScanBuilder &SetFilter(std::string_view filter) {
        try {
            ffi::scan_builder_set_filter(
                *impl_,
                rust::Slice<const uint8_t>(reinterpret_cast<const uint8_t *>(filter.data()), filter.size()));
        } catch (const rust::cxxbridge1::Error &e) {
            throw VortexException(e.what());
        }
        return *this;
    }

    /// Set the limit on the number of rows to scan.
    ScanBuilder &SetLimit(uint64_t limit) {
        ffi::scan_builder_set_limit(*impl_, limit);
        return *this;
    }

    // TODO(xinyu): In C++ API, do we want to return C DataInterface (only require nanoarrow as dep) or
    // RecordBatchReader (require arrow as dep)?
    /// Consume the scan builder to a stream of record batches.
    /// The scan builder is consumed and cannot be used after this call.
    arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> IntoStream();

    /// Consume the scan builder to an Arrow array and schema.
    /// The scan builder is consumed and cannot be used after this call.
    std::pair<ArrowArray, ArrowSchema> IntoArray();

private:
    friend class VortexFile;

    explicit ScanBuilder(rust::Box<ffi::VortexScanBuilder> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexScanBuilder> impl_;
};

class VortexWriteOptions {
public:
    VortexWriteOptions() : impl_(ffi::write_options_new()) {
    }

    /// Write an Arrow array stream to a Vortex file
    void WriteArrayStream(ArrowArrayStream &stream, const std::string &path) {
        try {
            ffi::write_array_stream(std::move(impl_), reinterpret_cast<uint8_t *>(&stream), path);
        } catch (const rust::cxxbridge1::Error &e) {
            throw VortexException(e.what());
        }
    }

private:
    rust::Box<ffi::VortexWriteOptions> impl_;
};

class VortexFile {
public:
    static VortexFile Open(const std::string &path) {
        try {
            return VortexFile(ffi::open_file(path));
        } catch (const rust::cxxbridge1::Error &e) {
            throw VortexException(e.what());
        }
    }

    /// Get the number of rows in the file.
    uint64_t RowCount() const {
        return ffi::file_row_count(*impl_);
    }

    /// Create a scan builder for the file.
    /// The scan builder can be used to scan the file.
    ScanBuilder CreateScanBuilder() const {
        return ScanBuilder(ffi::file_scan_builder(*impl_));
    }

private:
    explicit VortexFile(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexFile> impl_;
};

} // namespace vortex