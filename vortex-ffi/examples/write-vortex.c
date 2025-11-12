// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <OUTPUT_FILE>\n", argv[0]);
    printf("  Creates a Vortex file with example data demonstrating all array "
           "types.\n");
    return 1;
  }

  vx_error *error = NULL;
  vx_session *session = vx_session_new();
  if (session == NULL) {
    fprintf(stderr, "Failed to create Vortex session\n");
    return -1;
  }

  char *output_path = argv[1];
  printf("Creating Vortex file: %s\n\n", output_path);

  // ==========================================================================
  // Create example struct with various array types
  // ==========================================================================

  // 1. Create a primitive i32 array for "age" field
  printf("Creating i32 array (ages)...\n");
  int32_t ages_data[] = {25, 30, 35, 40, 45};
  bool ages_validity[] = {true, true, false, true, true}; // 3rd value is null
  size_t row_count = sizeof(ages_data) / sizeof(ages_data[0]);

  const vx_array *ages_array = vx_array_primitive_new_i32(
      ages_data, row_count, ages_validity, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to create ages array: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_session_free(session);
    return -1;
  }
  printf("  Created array with %zu rows\n", vx_array_len(ages_array));

  // 2. Create a UTF8 array for "name" field
  printf("Creating UTF8 array (names)...\n");
  vx_varbinview_builder *names_builder = vx_array_utf8_builder_new(false);

  const char *names[] = {"Alice", "Bob", "Charlie", "Diana", "Eve"};
  for (size_t i = 0; i < row_count; i++) {
    vx_varbinview_builder_append_utf8(names_builder, (const uint8_t *)names[i],
                                      strlen(names[i]), &error);
    if (error != NULL) {
      fprintf(stderr, "Failed to append UTF8 value: %s\n",
              vx_string_ptr(vx_error_get_message(error)));
      vx_error_free(error);
      vx_varbinview_builder_free(names_builder);
      vx_array_free(ages_array);
      vx_session_free(session);
      return -1;
    }
  }

  const vx_array *names_array = vx_varbinview_builder_finish(names_builder);
  printf("  Created array with %zu rows\n", vx_array_len(names_array));

  // 3. Create a bool array for "active" field
  printf("Creating bool array (active status)...\n");
  bool active_data[] = {true, false, true, true, false};
  const vx_array *active_array =
      vx_array_bool_new(active_data, row_count, NULL, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to create active array: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }
  printf("  Created array with %zu rows\n", vx_array_len(active_array));

  // 4. Create a float array for "score" field
  printf("Creating f64 array (scores)...\n");
  double scores_data[] = {98.5, 87.3, 92.1, 88.7, 95.2};
  const vx_array *scores_array =
      vx_array_primitive_new_f64(scores_data, row_count, NULL, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to create scores array: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_array_free(active_array);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }
  printf("  Created array with %zu rows\n", vx_array_len(scores_array));

  // 5. Create a decimal array for "salary" field
  printf("Creating decimal128 array (salaries)...\n");
  // Salaries in cents (e.g., 50000.00 = 5000000 cents)
  int128_t salaries_data[] = {5000000, 6000000, 7500000, 5500000, 8000000};
  uint8_t precision = 10;
  int8_t scale = 2; // 2 decimal places
  const vx_array *salaries_array = vx_array_decimal128_new(
      salaries_data, row_count, precision, scale, NULL, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to create salaries array: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_array_free(scores_array);
    vx_array_free(active_array);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }
  printf("  Created array with %zu rows\n", vx_array_len(salaries_array));

  // ==========================================================================
  // Build the struct dtype
  // ==========================================================================

  printf("\nBuilding struct dtype...\n");

  vx_struct_fields_builder *fields_builder = vx_struct_fields_builder_new();

  // Add field definitions
  const vx_string *name_str = vx_string_new("name");
  const vx_dtype *utf8_dtype = vx_dtype_utf8(false);
  vx_struct_fields_builder_add_field(fields_builder, name_str, utf8_dtype);

  const vx_string *age_str = vx_string_new("age");
  const vx_dtype *i32_dtype = vx_dtype_primitive(PTYPE_I32, true);
  vx_struct_fields_builder_add_field(fields_builder, age_str, i32_dtype);

  const vx_string *active_str = vx_string_new("active");
  const vx_dtype *bool_dtype = vx_dtype_bool(false);
  vx_struct_fields_builder_add_field(fields_builder, active_str, bool_dtype);

  const vx_string *score_str = vx_string_new("score");
  const vx_dtype *f64_dtype = vx_dtype_primitive(PTYPE_F64, false);
  vx_struct_fields_builder_add_field(fields_builder, score_str, f64_dtype);

  const vx_string *salary_str = vx_string_new("salary");
  const vx_dtype *decimal_dtype = vx_dtype_decimal(precision, scale, false);
  vx_struct_fields_builder_add_field(fields_builder, salary_str, decimal_dtype);

  const vx_struct_fields *struct_fields =
      vx_struct_fields_builder_finalize(fields_builder);
  const vx_dtype *struct_dtype = vx_dtype_struct(struct_fields, false);

  printf("  Struct dtype created with %llu fields\n",
         (unsigned long long)vx_struct_fields_nfields(struct_fields));

  // ==========================================================================
  // Create the struct array
  // ==========================================================================

  printf("\nCreating struct array...\n");

  // Array of field arrays (must match the order in struct_fields)
  const vx_array *field_arrays[] = {names_array, ages_array, active_array,
                                    scores_array, salaries_array};

  const vx_array *struct_array = vx_array_struct_new(
      struct_dtype, field_arrays, 5, row_count, NULL, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to create struct array: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_dtype_free(struct_dtype);
    vx_array_free(salaries_array);
    vx_array_free(scores_array);
    vx_array_free(active_array);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }
  printf("  Struct array created with %zu rows\n",
         vx_array_len(struct_array));

  // ==========================================================================
  // Write the array to file
  // ==========================================================================

  printf("\nWriting to file...\n");

  vx_array_sink *sink =
      vx_array_sink_open_file(session, output_path, struct_dtype, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to open file sink: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_array_free(struct_array);
    vx_dtype_free(struct_dtype);
    vx_array_free(salaries_array);
    vx_array_free(scores_array);
    vx_array_free(active_array);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }

  vx_array_sink_push(sink, struct_array, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to push array to sink: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_array_sink_free(sink);
    vx_array_free(struct_array);
    vx_dtype_free(struct_dtype);
    vx_array_free(salaries_array);
    vx_array_free(scores_array);
    vx_array_free(active_array);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }

  vx_array_sink_close(sink, &error);
  if (error != NULL) {
    fprintf(stderr, "Failed to close sink: %s\n",
            vx_string_ptr(vx_error_get_message(error)));
    vx_error_free(error);
    vx_array_sink_free(sink);
    vx_array_free(struct_array);
    vx_dtype_free(struct_dtype);
    vx_array_free(salaries_array);
    vx_array_free(scores_array);
    vx_array_free(active_array);
    vx_array_free(names_array);
    vx_array_free(ages_array);
    vx_session_free(session);
    return -1;
  }

  printf("  File written successfully!\n");

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  printf("\nCleaning up...\n");

  vx_array_sink_free(sink);
  vx_array_free(struct_array);
  vx_dtype_free(struct_dtype);
  // Note: field arrays are borrowed by struct_array, so we still need to free them
  vx_array_free(salaries_array);
  vx_array_free(scores_array);
  vx_array_free(active_array);
  vx_array_free(names_array);
  vx_array_free(ages_array);
  vx_session_free(session);

  printf("\nSuccess! Created Vortex file: %s\n", output_path);
  printf("  Contains %zu rows with 5 fields:\n", row_count);
  printf("    - name: UTF8\n");
  printf("    - age: i32 (nullable)\n");
  printf("    - active: bool\n");
  printf("    - score: f64\n");
  printf("    - salary: decimal(10,2)\n");
  printf("\nYou can read this file using the hello-vortex example:\n");
  printf("  ./hello_vortex %s\n", output_path);

  return 0;
}
