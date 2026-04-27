// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex.h"
#include <stdio.h>

const char *usage = "Print dtype of files\n"
                    "Usage: dtype <file glob>\n";

void print_dtype(const vx_dtype *dtype);

void print_ptype(const vx_dtype *type) {
    const char *ptype = NULL;

    switch (vx_dtype_primitive_ptype(type)) {
    case PTYPE_U8:
        ptype = "uint8_t";
        break;
    case PTYPE_U16:
        ptype = "uint16_t";
        break;
    case PTYPE_U32:
        ptype = "uint32_t";
        break;
    case PTYPE_U64:
        ptype = "uint64_t";
        break;
    case PTYPE_I8:
        ptype = "int8_t";
        break;
    case PTYPE_I16:
        ptype = "int16_t";
        break;
    case PTYPE_I32:
        ptype = "int32_t";
        break;
    case PTYPE_I64:
        ptype = "int64_t";
        break;
    case PTYPE_F16:
        ptype = "float16";
        break;
    case PTYPE_F32:
        ptype = "float";
        break;
    case PTYPE_F64:
        ptype = "double";
        break;
    default:
        __builtin_unreachable();
    }

    printf("primitive(%s)", ptype);
}

void print_struct_dtype(const vx_dtype *dtype) {
    const vx_struct_fields *fields = vx_dtype_struct_dtype(dtype);

    printf("struct(\n");
    for (uint64_t i = 0; i < vx_struct_fields_nfields(fields); ++i) {
        const vx_dtype *field_dtype = vx_struct_fields_field_dtype(fields, i);
        const vx_string *field_name = vx_struct_fields_field_name(fields, i);
        printf("    %.*s = ", (int)vx_string_len(field_name), vx_string_ptr(field_name));
        print_dtype(field_dtype);
        vx_dtype_free(field_dtype);
    }
    printf(")");
}

void print_list_dtype(const vx_dtype *dtype) {
    printf("list(");
    print_dtype(vx_dtype_list_element(dtype));
    printf(")");
}

void print_fixed_list_dtype(const vx_dtype *dtype) {
    printf("fixed list(size=%d, ", vx_dtype_fixed_size_list_size(dtype));
    print_dtype(vx_dtype_fixed_size_list_element(dtype));
    printf(")");
}

void print_decimal_dtype(const vx_dtype *dtype) {
    const uint8_t precision = vx_dtype_decimal_precision(dtype);
    const int8_t scale = vx_dtype_decimal_scale(dtype);
    printf("decimal(precision=%u, scale=%d)", precision, scale);
}

void print_dtype(const vx_dtype *dtype) {
    switch (vx_dtype_get_variant(dtype)) {
    case DTYPE_NULL:
        printf("null");
        break;
    case DTYPE_BOOL:
        printf("bool");
        break;
    case DTYPE_UTF8:
        printf("utf8");
        break;
    case DTYPE_BINARY:
        printf("binary");
        break;
    case DTYPE_EXTENSION:
        printf("extension");
        break;
    case DTYPE_PRIMITIVE:
        print_ptype(dtype);
        break;
    case DTYPE_STRUCT:
        print_struct_dtype(dtype);
        break;
    case DTYPE_LIST:
        print_list_dtype(dtype);
        break;
    case DTYPE_FIXED_SIZE_LIST:
        print_fixed_list_dtype(dtype);
        break;
    case DTYPE_DECIMAL:
        print_decimal_dtype(dtype);
        break;
    }
    printf("%c\n", vx_dtype_is_nullable(dtype) ? '?' : ' ');
}

void print_error(const char *what, const vx_error *error) {
    const vx_string *str = vx_error_get_message(error);
    fprintf(stderr, "%s: %.*s\n", what, (int)vx_string_len(str), vx_string_ptr(str));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "%s", usage);
        return 1;
    }

    vx_error *error = NULL;
    vx_session *const session = vx_session_new();
    if (session == NULL) {
        fprintf(stderr, "Failed to create Vortex session\n");
        return 1;
    }

    vx_data_source_options ds_options = {.paths = argv[1]};
    const vx_data_source *data_source = vx_data_source_new(session, &ds_options, &error);
    if (data_source == NULL) {
        print_error("Failed to create data source", error);
        vx_error_free(error);
        vx_session_free(session);
        return 1;
    }

    printf("dtype: ");
    print_dtype(vx_data_source_dtype(data_source));

    vx_data_source_free(data_source);
    vx_session_free(session);
    return 0;
}
