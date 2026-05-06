// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "nanoarrow/common/inline_types.h"
#include "nanoarrow/nanoarrow.h"

#define USE_OWN_ARROW
typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#include "vortex.h"

#include <stdio.h>
#include <unistd.h>

const char *usage = "Scan vortex files to Arrow\n"
                    "Usage: scan_to_arrow <file glob>\n";

void print_error(const char *what, const vx_error *error) {
    const vx_string *str = vx_error_get_message(error);
    fprintf(stderr, "%s: %.*s\n", what, (int)vx_string_len(str), vx_string_ptr(str));
}

void execute_scan(vx_session *session, vx_scan *scan) {
    vx_error *error = NULL;

    // Returned dtype is owned and mustn't be freed
    const vx_dtype *dtype = vx_scan_dtype(scan, &error);
    if (dtype == NULL) {
        print_error("Failed to get scan dtype", error);
        vx_error_free(error);
        return;
    }

    struct ArrowSchema schema;
    if (vx_dtype_to_arrow_schema(dtype, &schema, &error)) {
        print_error("Failed to convert dtype to Arrow schema", error);
        vx_error_free(error);
        return;
    }

    char schema_buf[1024 * 10];
    const int schema_len = ArrowSchemaToString(&schema, schema_buf, sizeof schema_buf, 1);
    printf("arrow schema: %.*s\n", schema_len, schema_buf);
    if (schema.release) {
        schema.release(&schema);
    }

    struct ArrowError arrow_error;
    ArrowErrorInit(&arrow_error);

    vx_partition *partition;
    size_t partitions = 0, arrays = 0, rows = 0;

    while ((partition = vx_scan_next_partition(scan, &error)) != NULL) {
        struct ArrowArrayStream stream;
        // Partition is consumed, we must not free it or use it after
        if (vx_partition_scan_arrow(session, partition, &stream, &error)) {
            print_error("Failed to scan partition to Arrow", error);
            vx_error_free(error);
            error = NULL;
            break;
        }

        struct ArrowArray array = {0};
        while (ArrowArrayStreamGetNext(&stream, &array, &arrow_error) == NANOARROW_OK &&
               array.release != NULL) {
            rows += array.length;
            ++arrays;
            array.release(&array);
            memset(&array, 0, sizeof(array));
        }

        printf("Read Partition %lu to arrow, %lu arrays, %lu rows\n", partitions, arrays, rows);
        rows = 0;
        arrays = 0;

        stream.release(&stream);
        ++partitions;
    }

    if (error) {
        print_error("Failed scanning partition", error);
        vx_error_free(error);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "%s", usage);
        return 1;
    }
    const char *paths = argv[1];

    vx_session *session = vx_session_new();
    if (session == NULL) {
        fprintf(stderr, "Failed to create Vortex session\n");
        return -1;
    }

    vx_data_source_options ds_options = {.paths = paths};
    vx_error *error = NULL;
    const vx_data_source *data_source = vx_data_source_new(session, &ds_options, &error);
    if (data_source == NULL) {
        print_error("Failed to create data source", error);
        // Returned errors are owned and need to be freed
        vx_error_free(error);
        vx_session_free(session);
        return 1;
    }

    vx_scan *scan = vx_data_source_scan(data_source, NULL, NULL, &error);
    if (scan == NULL) {
        print_error("Failed to create scan", error);
        vx_error_free(error);
        vx_data_source_free(data_source);
        vx_session_free(session);
        return 1;
    }

    execute_scan(session, scan);

    vx_scan_free(scan);
    vx_data_source_free(data_source);
    vx_session_free(session);

    return 0;
}
