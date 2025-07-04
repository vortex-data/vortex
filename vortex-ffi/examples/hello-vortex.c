// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <stdbool.h>
#include <stdio.h>
#include "vortex.h"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <VORTEX_FILE_URI>\n", argv[0]);
        return 1;
    }

    vx_error *error = NULL;
    vx_session *session = vx_session_new();
    if (session == NULL)
    {
        fprintf(stderr, "vx_session_new return NULL\n");
        return -1;
    }

    // Open the file
    char *path = argv[1];

    printf("Opening file %s\n", path);

    vx_file_open_options open_opts = {
        .uri = path,
        .property_keys = NULL,
        .property_vals = NULL,
        .property_len = 0,
    };
    printf("Scanning file: %s\n", path);

    const vx_file *file = vx_file_open_reader(&open_opts, session, &error);
    if (error != NULL)
    {
        fprintf(stderr, "error opening file\n");

        return -1;
    }

    // Start scanning, read new rows.
    vx_array_iterator *scan = vx_file_scan(file, NULL, &error);

    int chunk = 0;
    const vx_array *batch = vx_array_iterator_next(scan, &error);
    while (batch != NULL)
    {
        size_t len = vx_array_len(batch);
        printf("chunk %d has length %ld\n", chunk++, len);

        const vx_dtype *dtype = vx_array_dtype(batch);
        const vx_struct_fields *struct_fields = vx_dtype_struct_dtype(dtype);
        printf("Array has %zu fields\n", vx_struct_fields_nfields(struct_fields));
        vx_struct_fields_free(struct_fields);

        // free the batch
        vx_array_free(batch);
        // grab the next batch.
        batch = vx_array_iterator_next(scan, &error);
    }

    vx_array_iterator_free(scan);
    vx_file_free(file);

    if (error != NULL)
    {
        fprintf(stderr, "failed in scan operation\n");
        vx_session_free(session);
        vx_error_free(error);
        return -1;
    }

    printf("Scanning completed successfully\n");

    vx_session_free(session);
    return 0;
}
