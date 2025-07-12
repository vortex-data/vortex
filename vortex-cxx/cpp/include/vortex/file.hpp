// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <nanoarrow/common/inline_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace vortex {

class ScanBuilder {
public:
    ScanBuilder(ScanBuilder &&other) noexcept;
    ScanBuilder &operator=(ScanBuilder &&other) noexcept;
    ~ScanBuilder();

    ScanBuilder(const ScanBuilder &) = delete;
    ScanBuilder &operator=(const ScanBuilder &) = delete;

    /// Set the filter on the scan builder.
    ScanBuilder &SetFilter(std::string_view filter);

    /// Set the limit on the number of rows to scan.
    ScanBuilder &SetLimit(uint64_t limit);

    /// Consume the scan builder to a stream of record batches.
    /// The scan builder is consumed and cannot be used after this call.
    ArrowArrayStream IntoStream();

    /// Consume the scan builder to an Arrow array and schema.
    /// The scan builder is consumed and cannot be used after this call.
    std::pair<ArrowArray, ArrowSchema> IntoArray();

private:
    friend class VortexFile;

    struct Impl;
    explicit ScanBuilder(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

class VortexFile {
public:
    static VortexFile Open(const std::string &path);

    VortexFile(VortexFile &&other) noexcept;
    VortexFile &operator=(VortexFile &&other) noexcept;
    ~VortexFile();

    VortexFile(const VortexFile &) = delete;
    VortexFile &operator=(const VortexFile &) = delete;

    /// Get the number of rows in the file.
    uint64_t RowCount() const;

    /// Create a scan builder for the file.
    /// The scan builder can be used to scan the file.
    ScanBuilder CreateScanBuilder() const;

private:
    struct Impl;
    explicit VortexFile(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace vortex