// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex.hpp"

namespace vortex {

std::pair<ArrowArray, ArrowSchema> VortexFile::scan_to_arrow() const {
    try {
        auto arrow_c_structs = ffi::file_scan_to_arrow(*impl_);

        ArrowArray array;
        array.length = arrow_c_structs.array.length;
        array.null_count = arrow_c_structs.array.null_count;
        array.offset = arrow_c_structs.array.offset;
        array.n_buffers = arrow_c_structs.array.n_buffers;
        array.n_children = arrow_c_structs.array.n_children;
        array.buffers = reinterpret_cast<const void **>(arrow_c_structs.array.buffers);
        array.children = reinterpret_cast<struct ArrowArray **>(arrow_c_structs.array.children);
        array.dictionary = reinterpret_cast<struct ArrowArray *>(arrow_c_structs.array.dictionary);
        array.release = reinterpret_cast<void (*)(struct ArrowArray *)>(arrow_c_structs.array.release);
        array.private_data = reinterpret_cast<void *>(arrow_c_structs.array.private_data);

        ArrowSchema schema;
        schema.format = reinterpret_cast<const char *>(arrow_c_structs.schema.format);
        schema.name = reinterpret_cast<const char *>(arrow_c_structs.schema.name);
        schema.metadata = reinterpret_cast<const char *>(arrow_c_structs.schema.metadata);
        schema.flags = arrow_c_structs.schema.flags;
        schema.n_children = arrow_c_structs.schema.n_children;
        schema.children = reinterpret_cast<struct ArrowSchema **>(arrow_c_structs.schema.children);
        schema.dictionary = reinterpret_cast<struct ArrowSchema *>(arrow_c_structs.schema.dictionary);
        schema.release = reinterpret_cast<void (*)(struct ArrowSchema *)>(arrow_c_structs.schema.release);
        schema.private_data = reinterpret_cast<void *>(arrow_c_structs.schema.private_data);

        return {array, schema};
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> VortexFile::scan_to_stream() const {
    try {
        auto c_stream = ffi::file_scan_to_stream(*impl_);

        ArrowArrayStream stream;
        stream.get_schema =
            reinterpret_cast<int (*)(struct ArrowArrayStream *, struct ArrowSchema *)>(c_stream.get_schema);
        stream.get_next =
            reinterpret_cast<int (*)(struct ArrowArrayStream *, struct ArrowArray *)>(c_stream.get_next);
        stream.get_last_error =
            reinterpret_cast<const char *(*)(struct ArrowArrayStream *)>(c_stream.get_last_error);
        stream.release = reinterpret_cast<void (*)(struct ArrowArrayStream *)>(c_stream.release);
        stream.private_data = reinterpret_cast<void *>(c_stream.private_data);

        // Use Arrow C++ API to import the RecordBatchReader
        return arrow::ImportRecordBatchReader(&stream);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

ScanBuilder VortexFile::scan_builder() const {
    return ScanBuilder(ffi::file_scan_builder(*impl_));
}

} // namespace vortex