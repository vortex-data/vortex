// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <nanoarrow/common/inline_types.h>

#include <cstdint>
#include <memory>

namespace vortex {

class ScanBuilder {
public:
    ScanBuilder(ScanBuilder &&other) noexcept;
    ScanBuilder &operator=(ScanBuilder &&other) noexcept;
    ~ScanBuilder();

    ScanBuilder(const ScanBuilder &) = delete;
    ScanBuilder &operator=(const ScanBuilder &) = delete;

    /// Only include rows in the range [row_range_start, row_range_end).
    ScanBuilder &WithRowRange(uint64_t row_range_start, uint64_t row_range_end);

    /// Only include rows with the given indices.
    ScanBuilder &WithIncludeByIndex(const uint64_t *indices, std::size_t size);

    /// Set the limit on the number of rows to scan out.
    ScanBuilder &WithLimit(uint64_t limit);

    /// Set the output schema on the scan builder.
    /// TODO: should decide to input full schema or schema after adding projection.
    ScanBuilder &WithOutputSchema(ArrowSchema &output_schema);

    /// Take ownership and consume the scan builder to a stream of record batches.
    ArrowArrayStream IntoStream();

private:
    friend class VortexFile;

    struct Impl;
    explicit ScanBuilder(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};
} // namespace vortex