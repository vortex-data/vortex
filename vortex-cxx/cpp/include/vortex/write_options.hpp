// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cstddef>
#include <cstdint>
#include <nanoarrow/common/inline_types.h>

#include "vortex/session.hpp"
#include "vortex/writer.hpp"
#include "vortex_cxx_bridge/lib.h"

namespace vortex {

enum class FileStat : uint8_t {
    /// Whether all values are the same (nulls are not equal to other non-null values,
    /// so this is true iff all values are null or all values are the same non-null value)
    IsConstant = 0,
    /// Whether the non-null values in the array are sorted in ascending order (i.e., we skip nulls)
    /// This may later be extended to support descending order, but for now we only support ascending order.
    IsSorted = 1,
    /// Whether the non-null values in the array are strictly sorted in ascending order (i.e., sorted with no
    /// duplicates)
    /// This may later be extended to support descending order, but for now we only support ascending order.
    IsStrictSorted = 2,
    /// The maximum value in the array (ignoring nulls, unless all values are null)
    Max = 3,
    /// The minimum value in the array (ignoring nulls, unless all values are null)
    Min = 4,
    /// The sum of the non-null values of the array.
    Sum = 5,
    /// The number of null values in the array
    NullCount = 6,
    /// The uncompressed size of the array in bytes
    UncompressedSizeInBytes = 7,
    /// The number of NaN values in the array
    NaNCount = 8,
};

class VortexWriteStrategy {
public:
    VortexWriteStrategy(VortexWriteStrategy &&other) noexcept = default;
    VortexWriteStrategy &operator=(VortexWriteStrategy &&other) noexcept = default;
    ~VortexWriteStrategy() = default;

    VortexWriteStrategy(const VortexWriteStrategy &) = delete;
    VortexWriteStrategy &operator=(const VortexWriteStrategy &) = delete;

private:
    friend class VortexWriteOptions;
    friend class VortexWriteStrategyBuilder;

    explicit VortexWriteStrategy(rust::Box<ffi::VortexWriteStrategy> impl) : impl_(std::move(impl)) {
    }

    const rust::Box<ffi::VortexWriteStrategy> &GetImpl() const {
        return impl_;
    }

    rust::Box<ffi::VortexWriteStrategy> impl_;
};

/// Note: Some of features are not exposed yet
class VortexWriteStrategyBuilder {
public:
    VortexWriteStrategyBuilder() : impl_(ffi::write_strategy_builder_new()) {
    }
    VortexWriteStrategyBuilder(VortexWriteStrategyBuilder &&other) noexcept = default;
    VortexWriteStrategyBuilder &operator=(VortexWriteStrategyBuilder &&other) noexcept = default;
    ~VortexWriteStrategyBuilder() = default;

    VortexWriteStrategyBuilder(const VortexWriteStrategyBuilder &) = delete;
    VortexWriteStrategyBuilder &operator=(const VortexWriteStrategyBuilder &) = delete;

    /// Configure compact encodings.
    VortexWriteStrategyBuilder &WithCompactEncodings();

    /// Set row block size.
    VortexWriteStrategyBuilder &WithRowBlockSize(std::size_t row_block_size);

    /// Build and consume strategy. 
    VortexWriteStrategy Build();

private:
    rust::Box<ffi::VortexWriteStrategyBuilder> impl_;
};

class VortexWriteOptions {
public:
    VortexWriteOptions() : impl_(ffi::write_options_new()) {
    }
    explicit VortexWriteOptions(const VortexSession &session)
        : impl_(ffi::write_options_new_with_session(*session.GetImpl())) {
    }
    VortexWriteOptions(VortexWriteOptions &&other) noexcept = default;
    VortexWriteOptions &operator=(VortexWriteOptions &&other) noexcept = default;
    ~VortexWriteOptions() = default;

    VortexWriteOptions(const VortexWriteOptions &) = delete;
    VortexWriteOptions &operator=(const VortexWriteOptions &) = delete;

    /// Write an ArrowArrayStream to a Vortex file
    void WriteArrayStream(ArrowArrayStream &stream, const std::string &path);

    /// Disable file statistics.
    VortexWriteOptions &WithoutFileStatistics();

    /// Configure which file statistics are used.
    VortexWriteOptions &WithFileStatistics(const FileStat *statistics, std::size_t size);

    /// Exclude DType from file footer.
    VortexWriteOptions &ExcludeDType();

    /// Apply strategy.
    VortexWriteOptions &WithStrategy(const VortexWriteStrategy &strategy);

    VortexWriter CreateWriter(const std::string &path);

private:
    rust::Box<ffi::VortexWriteOptions> impl_;
};

} // namespace vortex
