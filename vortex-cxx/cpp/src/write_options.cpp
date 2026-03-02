// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/write_options.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"

namespace vortex {
VortexWriteStrategyBuilder &VortexWriteStrategyBuilder::WithCompactEncodings() {
    try {
        ffi::write_strategy_builder_with_compact_encodings(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

VortexWriteStrategyBuilder &VortexWriteStrategyBuilder::WithRowBlockSize(std::size_t row_block_size) {
    try {
        ffi::write_strategy_builder_with_row_block_size(*impl_, row_block_size);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

VortexWriteStrategy VortexWriteStrategyBuilder::Build() {
    try {
        return VortexWriteStrategy(ffi::write_strategy_builder_build(std::move(impl_)));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

void VortexWriteOptions::WriteArrayStream(ArrowArrayStream &stream, const std::string &path) {
    try {
        ffi::write_array_stream(std::move(impl_), reinterpret_cast<uint8_t *>(&stream), path);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

VortexWriteOptions &VortexWriteOptions::WithoutFileStatistics() {
    ffi::write_options_without_file_statistics(*impl_);
    return *this;
}

VortexWriteOptions &VortexWriteOptions::WithFileStatistics(const FileStat *statistics, std::size_t size) {
    try {
        rust::Slice<const ffi::FileStat> ffi_stats(reinterpret_cast<const ffi::FileStat *>(statistics), size);
        ffi::write_options_with_file_statistics(*impl_, ffi_stats);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

VortexWriteOptions &VortexWriteOptions::ExcludeDType() {
    ffi::write_options_exclude_dtype(*impl_);
    return *this;
}

VortexWriteOptions &VortexWriteOptions::WithStrategy(const VortexWriteStrategy &strategy) {
    ffi::write_options_with_strategy(*impl_, *strategy.GetImpl());
    return *this;
}

VortexWriter VortexWriteOptions::CreateWriter(const std::string &path) {
    try {
        return VortexWriter(ffi::write_options_into_writer(std::move(impl_), path));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

} // namespace vortex
