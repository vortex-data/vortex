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

class VortexFile {
public:
    static VortexFile open(const std::string &path) {
        try {
            return VortexFile(ffi::open_file(path));
        } catch (const rust::cxxbridge1::Error &e) {
            throw VortexException(e.what());
        }
    }

    explicit VortexFile(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {
    }

    uint64_t row_count() const {
        return ffi::file_row_count(*impl_);
    }

    std::pair<ArrowArray, ArrowSchema> scan_to_arrow() const;

    arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> scan_to_stream() const;

private:
    rust::Box<ffi::VortexFile> impl_;
};

} // namespace vortex