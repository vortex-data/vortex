// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SAMPLE_ROWS 200

const char *usage = "Write a sample 200 rows .vortex file\n"
                    "Usage: write_sample <output file path>\n";

// StructArray { age=u8, height=u16? }
const vx_dtype *sample_dtype(void) {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();

    const char *age = "age";
    const vx_string *age_name = vx_string_new(age, strlen(age));
    const vx_dtype *age_type = vx_dtype_new_primitive(PTYPE_U8, false);
    vx_struct_fields_builder_add_field(builder, age_name, age_type);

    const char *height = "height";
    const vx_string *height_name = vx_string_new(height, strlen(height));
    const vx_dtype *height_type = vx_dtype_new_primitive(PTYPE_U16, true);
    vx_struct_fields_builder_add_field(builder, height_name, height_type);

    vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    return vx_dtype_new_struct(fields, false);
}

void print_error(const char *what, const vx_error *error) {
    const vx_string *str = vx_error_get_message(error);
    fprintf(stderr, "%s: %.*s\n", what, (int)vx_string_len(str), vx_string_ptr(str));
}

const vx_array *sample_array(void) {
    vx_validity validity = {.type = VX_VALIDITY_NON_NULLABLE};
    vx_struct_column_builder *builder = vx_struct_column_builder_new(&validity, SAMPLE_ROWS);

    uint8_t age_buffer[SAMPLE_ROWS];
    uint16_t height_buffer[SAMPLE_ROWS];
    for (uint8_t i = 0; i < SAMPLE_ROWS; ++i) {
        age_buffer[i] = i;
        height_buffer[i] = rand() % (i + 1);
    }

    vx_error *error = NULL;
    const vx_array *age_array = vx_array_new_primitive(PTYPE_U8, age_buffer, SAMPLE_ROWS, &validity, &error);
    if (error != NULL) {
        print_error("Error creating age array", error);
        vx_error_free(error);
        vx_struct_column_builder_free(builder);
        return NULL;
    }

    vx_struct_column_builder_add_field(builder, "age", age_array, &error);
    vx_array_free(age_array);
    if (error != NULL) {
        print_error("Error adding age array field to root array", error);
        vx_error_free(error);
        vx_struct_column_builder_free(builder);
        return NULL;
    }

    validity.type = VX_VALIDITY_ALL_VALID;
    const vx_array *height_array =
        vx_array_new_primitive(PTYPE_U16, height_buffer, SAMPLE_ROWS, &validity, &error);
    if (error != NULL) {
        print_error("Error adding height array field to root array", error);
        vx_error_free(error);
        vx_struct_column_builder_free(builder);
        return NULL;
    }

    vx_struct_column_builder_add_field(builder, "height", height_array, &error);
    vx_array_free(height_array);
    if (error != NULL) {
        print_error("Error adding height array field to root array", error);
        vx_error_free(error);
        vx_struct_column_builder_free(builder);
        return NULL;
    }

    const vx_array *array = vx_struct_column_builder_finalize(builder, &error);
    if (error != NULL) {
        print_error("Error creating struct array", error);
        vx_error_free(error);
        return NULL;
    }

    return array;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "%s", usage);
        return 1;
    }
    const char *output = argv[1];

    vx_session *const session = vx_session_new();
    if (session == NULL) {
        fprintf(stderr, "Failed to create Vortex session\n");
        return 1;
    }

    const vx_dtype *dtype = sample_dtype();

    vx_error *error = NULL;
    vx_array_sink *sink = vx_array_sink_open_file(session, output, dtype, &error);

    vx_dtype_free(dtype);
    if (error != NULL) {
        vx_session_free(session);
        return 1;
    }

    const vx_array *array = sample_array();
    if (array == NULL) {
        // We already have an error, so we can ignore a potential error
        // from this operation
        vx_array_sink_close(sink, &error);
        vx_session_free(session);
        return 1;
    }

    vx_array_sink_push(sink, array, &error);
    if (error != NULL) {
        vx_array_sink_close(sink, &error);
        vx_session_free(session);
        return 1;
    }
    vx_array_free(array);

    vx_array_sink_close(sink, &error);
    if (error != NULL) {
        print_error("Error closing output sink", error);
        vx_error_free(error);
    }

    vx_session_free(session);
    return 0;
}
