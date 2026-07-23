// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <nanoarrow/nanoarrow.h>

typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#define USE_OWN_ARROW 1

#include <cstring>
#include <iostream>
#include <string_view>
#include <vortex/data_source.hpp>

using vortex::ArrowStream;
using vortex::DataSource;
using vortex::DataType;
using vortex::Scan;
using vortex::Session;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Scan vortex files to Arrow\nUsage: scan_to_arrow <file glob>\n";
        return 1;
    }
    const char *paths = argv[1];

    Session session;
    DataSource ds = DataSource::open(session, {paths});
    Scan scan = ds.scan();

    DataType out_dtype = ds.dtype();
    ArrowSchema schema = out_dtype.to_arrow();
    char schema_buf[10 * 1024];
    const int64_t schema_len = ArrowSchemaToString(&schema, schema_buf, sizeof schema_buf, 1);
    std::cout << "arrow schema: " << std::string_view {schema_buf, static_cast<size_t>(schema_len)} << '\n';
    if (schema.release != nullptr) {
        schema.release(&schema);
    }

    ArrowError arrow_error;
    ArrowErrorInit(&arrow_error);

    size_t partition_idx = 0;
    while (auto partition = scan.next_partition()) {
        ArrowStream stream = std::move(*partition).into_arrow_stream();

        size_t rows = 0;
        size_t array_count = 0;
        ArrowArray array = {};
        while (ArrowArrayStreamGetNext(stream.raw(), &array, &arrow_error) == NANOARROW_OK &&
               array.release != nullptr) {
            rows += array.length;
            ++array_count;
            array.release(&array);
            std::memset(&array, 0, sizeof(array));
        }
        std::cout << "Read partition " << partition_idx << " to Arrow, " << array_count << " arrays, " << rows
                  << " rows\n";
        ++partition_idx;
    }
    return 0;
}
