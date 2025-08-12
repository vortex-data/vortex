// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "vortex.h"

// Function declaration for runtime shutdowns
extern bool vx_runtime_shutdown();

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
    printf("Opening file: %s\n", path);

    vx_file_open_options open_opts = {
        .uri = path,
        .property_keys = NULL,
        .property_vals = NULL,
        .property_len = 0,
    };

    const vx_file *file = vx_file_open_reader(&open_opts, session, &error);
    if (error != NULL)
    {
        fprintf(stderr, "Error opening file\n");
        vx_session_free(session);
        vx_error_free(error);
        vx_runtime_shutdown();
        return -1;
    }

    // Print file metadata
    uint64_t row_count = vx_file_row_count(file);
    printf("File contains %llu total rows\n", (unsigned long long)row_count);

    // Get and display file dtype
    const vx_dtype *file_dtype = vx_file_dtype(file);
    vx_dtype_variant variant = vx_dtype_get_variant(file_dtype);
    bool nullable = vx_dtype_is_nullable(file_dtype);
    printf("File DType variant: %d, nullable: %s\n", variant, nullable ? "true" : "false");

    // Start scanning
    printf("\nScanning file...\n");
    vx_array_iterator *scan = vx_file_scan(file, NULL, &error);
    if (error != NULL)
    {
        fprintf(stderr, "Error creating scan\n");
        vx_file_free(file);
        vx_error_free(error);
        vx_session_free(session);
        vx_runtime_shutdown();
        return -1;
    }

    int chunk_count = 0;
    int max_chunks_to_show = 3; // Limit detailed output to first 3 chunks
    const vx_array *batch = vx_array_iterator_next(scan, &error);

    while (batch != NULL && error == NULL)
    {
        size_t batch_len = vx_array_len(batch);
        printf("\nChunk %d: %zu rows\n", chunk_count, batch_len);

        // For the first few chunks, show more details
        if (chunk_count < max_chunks_to_show && batch_len > 0)
        {
            const vx_dtype *dtype = vx_array_dtype(batch);
            vx_dtype_variant batch_variant = vx_dtype_get_variant(dtype);

            // Check null count (may fail for some array types)
            uint32_t null_count = vx_array_null_count(batch, &error);
            if (error == NULL)
            {
                printf("  Null count: %u\n", null_count);
            }
            else
            {
                // Clear error and continue
                vx_error_free(error);
                error = NULL;
            }

            // If it's a struct, show field information
            if (batch_variant == DTYPE_STRUCT)
            {
                const vx_struct_fields *fields = vx_dtype_struct_dtype(dtype);
                size_t n_fields = vx_struct_fields_nfields(fields);
                printf("  Struct with %zu fields:\n", n_fields);

                for (size_t i = 0; i < n_fields && i < 5; i++) // Show up to 5 fields
                {
                    const vx_string *field_name = vx_struct_fields_field_name(fields, i);
                    const vx_dtype *field_dtype = vx_struct_fields_field_dtype(fields, i);

                    size_t name_len = vx_string_len(field_name);
                    const char *name_ptr = vx_string_ptr(field_name);
                    printf("    Field %zu: %.*s", i, (int)name_len, name_ptr);

                    vx_dtype_variant field_variant = vx_dtype_get_variant(field_dtype);
                    if (field_variant == DTYPE_PRIMITIVE)
                    {
                        vx_ptype ptype = vx_dtype_primitive_ptype(field_dtype);
                        printf(" (Primitive type %d)\n", ptype);
                    }
                    else
                    {
                        printf(" (Type variant %d)\n", field_variant);
                    }

                    // For first chunk, also test field array access
                    if (chunk_count == 0 && i == 0)
                    {
                        const vx_array *field_array = vx_array_get_field(batch, i, &error);
                        if (error == NULL && field_array != NULL)
                        {
                            size_t field_len = vx_array_len(field_array);
                            printf("      Field array length: %zu\n", field_len);

                            // Try to slice the field array (first 5 elements)
                            if (field_len > 5)
                            {
                                const vx_array *sliced = vx_array_slice(field_array, 0, 5, &error);
                                if (error == NULL && sliced != NULL)
                                {
                                    printf("      Successfully sliced first 5 elements\n");
                                    vx_array_free(sliced);
                                }
                                else if (error != NULL)
                                {
                                    vx_error_free(error);
                                    error = NULL;
                                }
                            }

                            vx_array_free(field_array);
                        }
                        else if (error != NULL)
                        {
                            vx_error_free(error);
                            error = NULL;
                        }
                    }

                    vx_string_free(field_name);
                    vx_dtype_free(field_dtype);
                }
                vx_struct_fields_free(fields);
            }
            else
            {
                printf("  Batch DType variant: %d\n", batch_variant);
            }
        }

        vx_array_free(batch);
        batch = vx_array_iterator_next(scan, &error);
        chunk_count++;
    }

    printf("\nTotal chunks processed: %d\n", chunk_count);

    vx_array_iterator_free(scan);
    vx_file_free(file);

    if (error != NULL)
    {
        fprintf(stderr, "Error during scan operation\n");
        vx_error_free(error);
        vx_session_free(session);
        vx_runtime_shutdown();
        return -1;
    }

    printf("Scanning completed successfully\n");

    vx_session_free(session);

    // Explicitly shutdown the runtime to prevent cleanup race with mimalloc
    if (vx_runtime_shutdown())
    {
        printf("Runtime shutdown successfully\n");
    }
    else
    {
        printf("Runtime shutdown failed (may still have active references)\n");
    }

    return 0;
}