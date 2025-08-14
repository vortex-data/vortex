// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "vortex/expr.hpp"
#include <nanoarrow/common/inline_types.h>
#include "vortex_cxx_bridge/lib.h"

#include <cstdint>

namespace vortex {
/// The StreamDriver internally holds a `RecordBatchIteratorAdapter` from the Rust side, which is thread-safe
/// and cloneable. The `RecordBatchIteratorAdapter` internally holds a `WorkStealingArrayIterator`.
class StreamDriver {
public:
    StreamDriver(StreamDriver &&other) noexcept = default;
    StreamDriver &operator=(StreamDriver &&other) noexcept = default;
    ~StreamDriver() = default;

    StreamDriver(const StreamDriver &) = delete;
    StreamDriver &operator=(const StreamDriver &) = delete;

    /// Create a stream of record batches.
    ///
    /// This function is thread-safe and can be called from multiple threads to create one stream per
    /// thread to make progress on the same StreamDriver that is built from a ScanBuilder concurrently.
    ///
    /// Within each thread, the record batches will be emitted in the original order they are within
    /// the scan. Between threads, the order is not guaranteed.
    ///
    /// Example: If the scan contains batches [b0, b1, b2, b3, b4, b5] and two threads call this
    /// function respectively to make progress on their own stream, Thread 1 might receive [b0,
    /// b2, b4] and Thread 2 might receive [b1, b3, b5]. Each thread maintains order within its
    /// subset, but overall ordering between threads is not guaranteed (e.g., Thread 2 could emit b1
    /// before Thread 1 emits b0).
    ArrowArrayStream CreateArrayStream() const;

private:
    friend class ScanBuilder;

    explicit StreamDriver(rust::Box<ffi::ThreadsafeCloneableReader> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::ThreadsafeCloneableReader> impl_;
};

using expr::Expr;
class ScanBuilder {
public:
    ScanBuilder(ScanBuilder &&other) noexcept = default;
    ScanBuilder &operator=(ScanBuilder &&other) noexcept = default;
    ~ScanBuilder() = default;

    ScanBuilder(const ScanBuilder &) = delete;
    ScanBuilder &operator=(const ScanBuilder &) = delete;

    /// Only include rows that match the filter expressions.
    ScanBuilder &&WithFilter(Expr expr) &&;

    /// Only include columns that match the projection expressions.
    ScanBuilder &&WithProjection(Expr expr) &&;

    /// Only include rows in the range [row_range_start, row_range_end).
    ScanBuilder &&WithRowRange(uint64_t row_range_start, uint64_t row_range_end) &&;

    /// Only include rows with the given indices.
    ScanBuilder &&WithIncludeByIndex(const uint64_t *indices, std::size_t size) &&;

    /// Set the limit on the number of rows to scan out.
    ScanBuilder &&WithLimit(uint64_t limit) &&;

    /// Set the output schema on the scan builder.
    /// TODO: currently if pass in this option, the schema needs to be the schema after adding projection.
    ScanBuilder &&WithOutputSchema(ArrowSchema &output_schema) &&;

    /// Take ownership and consume the scan builder to a stream of record batches.
    ArrowArrayStream IntoStream() &&;

    /// Take ownership and consume the scan builder to a stream driver.
    /// Under the hood, this function calls `ScanBuilder::into_record_batch_reader` and holds a
    /// `WorkStealingArrayIterator` in StreamDriver.
    StreamDriver IntoStreamDriver() &&;

private:
    friend class VortexFile;

    explicit ScanBuilder(rust::Box<ffi::VortexScanBuilder> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexScanBuilder> impl_;
};
} // namespace vortex