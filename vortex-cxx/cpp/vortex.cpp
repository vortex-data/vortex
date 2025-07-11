// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex.hpp"

namespace vortex {
arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ScanBuilder::IntoStream() {
    try {
        ArrowArrayStream stream;
        ffi::scan_builder_into_stream(std::move(impl_), reinterpret_cast<uint8_t *>(&stream));
        return arrow::ImportRecordBatchReader(&stream);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

std::pair<ArrowArray, ArrowSchema> ScanBuilder::IntoArray() {
    try {
        ArrowArray array;
        ArrowSchema schema;
        ffi::scan_builder_into_arrow(std::move(impl_), reinterpret_cast<uint8_t *>(&array),
                                     reinterpret_cast<uint8_t *>(&schema));
        return {array, schema};
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}
} // namespace vortex