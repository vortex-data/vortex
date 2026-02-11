// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "common.h"

// Writer API for creating Vortex files from ClickHouse.
//
// Note: The writer API is still under development and may change.
//
// Usage (planned):
//   VortexWriter* writer = vortex_writer_new("/path/to/output.vortex");
//   if (writer == NULL) { /* handle error */ }
//
//   // Set schema
//   vortex_writer_set_schema(writer, schema);
//
//   // Write data
//   vortex_writer_write_batch(writer, columns, num_columns, num_rows);
//
//   // Finalize
//   vortex_writer_finalize(writer);
//   vortex_writer_free(writer);

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Writer Creation and Destruction
// =============================================================================

/// Create a new Vortex writer for the given output path.
///
/// @param path Null-terminated C string with the output file path.
/// @return Writer handle, or NULL on error.
VortexWriter* vortex_writer_new(const char* path);

/// Free a Vortex writer.
///
/// @param writer Writer handle to free. NULL is safely ignored.
void vortex_writer_free(VortexWriter* writer);

// =============================================================================
// Schema Configuration
// =============================================================================

/// Add a column to the writer's schema.
///
/// @param writer Writer handle.
/// @param name Column name (null-terminated).
/// @param clickhouse_type ClickHouse type string (e.g., "Int64", "String").
/// @param nullable Whether the column is nullable.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_add_column(
    VortexWriter* writer,
    const char* name,
    const char* clickhouse_type,
    int32_t nullable
);

// =============================================================================
// Data Writing
// =============================================================================

/// Write a batch of data (simplified API for primitive-only columns).
///
/// For columns that include string types, use the batch API:
/// vortex_writer_begin_batch(), vortex_writer_write_column_*(), vortex_writer_end_batch()
///
/// @param writer Writer handle.
/// @param data Array of pointers to column data.
/// @param num_columns Number of columns.
/// @param num_rows Number of rows in this batch.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_write_batch(
    VortexWriter* writer,
    const void* const* data,
    size_t num_columns,
    size_t num_rows
);

// =============================================================================
// Batch Writing API (supports mixed primitive and string columns)
// =============================================================================

/// Begin writing a new batch with the given number of rows.
///
/// After calling this, use vortex_writer_write_column() or
/// vortex_writer_write_string_column() to write each column's data,
/// then call vortex_writer_end_batch() to commit the batch.
///
/// @param writer Writer handle.
/// @param num_rows Number of rows in this batch.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_begin_batch(
    VortexWriter* writer,
    size_t num_rows
);

/// Write a primitive column by index.
///
/// Must be called between vortex_writer_begin_batch() and vortex_writer_end_batch().
///
/// @param writer Writer handle.
/// @param column_index Column index (0-based).
/// @param data Pointer to column data (array of the appropriate primitive type).
/// @param num_rows Number of rows (must match begin_batch).
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_write_column(
    VortexWriter* writer,
    size_t column_index,
    const void* data,
    size_t num_rows
);

/// Write a nullable primitive column by index with null map.
///
/// The null_map uses ClickHouse's convention: one byte per row where:
/// - 0 = valid (not null)
/// - 1 = null
///
/// Must be called between vortex_writer_begin_batch() and vortex_writer_end_batch().
///
/// @param writer Writer handle.
/// @param column_index Column index (0-based).
/// @param data Pointer to column data (array of the appropriate primitive type).
/// @param null_map Pointer to null map (array of num_rows bytes), or NULL for all-valid.
/// @param num_rows Number of rows (must match begin_batch).
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_write_column_nullable(
    VortexWriter* writer,
    size_t column_index,
    const void* data,
    const uint8_t* null_map,
    size_t num_rows
);

/// Write a string column by index.
///
/// The strings are provided as concatenated data with offsets.
/// The offsets array must have num_rows + 1 elements, where:
/// - offsets[i] is the start offset of string i
/// - offsets[num_rows] is the total data length
///
/// Must be called between vortex_writer_begin_batch() and vortex_writer_end_batch().
///
/// @param writer Writer handle.
/// @param column_index Column index (0-based).
/// @param data Pointer to concatenated string data.
/// @param offsets Pointer to offsets array (num_rows + 1 elements).
/// @param num_rows Number of rows (must match begin_batch).
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_write_string_column(
    VortexWriter* writer,
    size_t column_index,
    const uint8_t* data,
    const uint64_t* offsets,
    size_t num_rows
);

/// Write a nullable string column by index with null map.
///
/// The strings are provided as concatenated data with offsets.
/// The null_map uses ClickHouse's convention: one byte per row where:
/// - 0 = valid (not null)
/// - 1 = null
///
/// Must be called between vortex_writer_begin_batch() and vortex_writer_end_batch().
///
/// @param writer Writer handle.
/// @param column_index Column index (0-based).
/// @param data Pointer to concatenated string data.
/// @param offsets Pointer to offsets array (num_rows + 1 elements).
/// @param null_map Pointer to null map (array of num_rows bytes), or NULL for all-valid.
/// @param num_rows Number of rows (must match begin_batch).
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_write_string_column_nullable(
    VortexWriter* writer,
    size_t column_index,
    const uint8_t* data,
    const uint64_t* offsets,
    const uint8_t* null_map,
    size_t num_rows
);

/// End the current batch and commit it.
///
/// All columns must be written before calling this function.
///
/// @param writer Writer handle.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_end_batch(VortexWriter* writer);

// =============================================================================
// List (Array) Column Writing
// =============================================================================

/// Write the offsets for a List (Array) column.
///
/// After writing offsets, use vortex_writer_list_write_element_column() or
/// vortex_writer_list_write_element_string_column() to write the nested elements.
/// Finally, call vortex_writer_list_end() to commit the list column.
///
/// @param writer Writer handle.
/// @param column_index Column index (0-based) of the list column.
/// @param offsets Array of num_rows + 1 uint64_t offsets (start offset of each list element).
/// @param null_map Pointer to null map (one byte per row, 0=valid, 1=null), or NULL for all-valid.
/// @param num_rows Number of rows (must match begin_batch).
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_list_write_offsets(
    VortexWriter* writer,
    size_t column_index,
    const uint64_t* offsets,
    const uint8_t* null_map,
    size_t num_rows
);

/// Write primitive element data for a list column.
///
/// Must be called after vortex_writer_list_write_offsets().
/// The element count must equal offsets[num_rows] - offsets[0].
///
/// @param writer Writer handle.
/// @param column_index Column index of the list column.
/// @param data Pointer to element data.
/// @param num_elements Number of elements.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_list_write_element_column(
    VortexWriter* writer,
    size_t column_index,
    const void* data,
    size_t num_elements
);

/// Write string element data for a list column.
///
/// Must be called after vortex_writer_list_write_offsets().
///
/// @param writer Writer handle.
/// @param column_index Column index of the list column.
/// @param data Pointer to concatenated string data.
/// @param offsets Pointer to string offsets array (num_elements + 1).
/// @param num_elements Number of string elements.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_list_write_element_string_column(
    VortexWriter* writer,
    size_t column_index,
    const uint8_t* data,
    const uint64_t* offsets,
    size_t num_elements
);

/// Write nullable primitive element data for a list column.
///
/// Must be called after vortex_writer_list_write_offsets().
/// The null_map uses ClickHouse's convention: one byte per element where
/// 0 = valid (not null), 1 = null.
///
/// @param writer Writer handle.
/// @param column_index Column index of the list column.
/// @param data Pointer to element data.
/// @param null_map Pointer to null map (one byte per element), or NULL for all-valid.
/// @param num_elements Number of elements.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_list_write_element_column_nullable(
    VortexWriter* writer,
    size_t column_index,
    const void* data,
    const uint8_t* null_map,
    size_t num_elements
);

/// Write nullable string element data for a list column.
///
/// Must be called after vortex_writer_list_write_offsets().
/// The null_map uses ClickHouse's convention: one byte per element where
/// 0 = valid (not null), 1 = null.
///
/// @param writer Writer handle.
/// @param column_index Column index of the list column.
/// @param data Pointer to concatenated string data.
/// @param offsets Pointer to string offsets array (num_elements + 1).
/// @param null_map Pointer to null map (one byte per element), or NULL for all-valid.
/// @param num_elements Number of string elements.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_list_write_element_string_column_nullable(
    VortexWriter* writer,
    size_t column_index,
    const uint8_t* data,
    const uint64_t* offsets,
    const uint8_t* null_map,
    size_t num_elements
);

/// Finalize the list column data.
///
/// Must be called after element data is written.
///
/// @param writer Writer handle.
/// @param column_index Column index of the list column.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_list_end(
    VortexWriter* writer,
    size_t column_index
);

// =============================================================================
// Struct (Tuple) Column Writing
// =============================================================================

/// Begin writing a struct (Tuple) column.
///
/// After calling this, use vortex_writer_struct_write_field() or
/// vortex_writer_struct_write_field_string() to write each field.
/// Then call vortex_writer_struct_end() to commit the struct column.
///
/// @param writer Writer handle.
/// @param column_index Column index of the struct column.
/// @param null_map Pointer to null map (one byte per row, 0=valid, 1=null), or NULL for all-valid.
/// @param num_rows Number of rows.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_struct_begin(
    VortexWriter* writer,
    size_t column_index,
    const uint8_t* null_map,
    size_t num_rows
);

/// Write a primitive field of a struct column.
///
/// @param writer Writer handle.
/// @param column_index Column index of the struct column.
/// @param field_index Field index within the struct (0-based).
/// @param data Pointer to field data.
/// @param null_map Pointer to null map for this field, or NULL for all-valid.
/// @param num_rows Number of rows.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_struct_write_field(
    VortexWriter* writer,
    size_t column_index,
    size_t field_index,
    const void* data,
    const uint8_t* null_map,
    size_t num_rows
);

/// Write a string field of a struct column.
///
/// @param writer Writer handle.
/// @param column_index Column index of the struct column.
/// @param field_index Field index within the struct (0-based).
/// @param data Pointer to concatenated string data.
/// @param offsets Pointer to string offsets array (num_rows + 1).
/// @param null_map Pointer to null map for this field, or NULL for all-valid.
/// @param num_rows Number of rows.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_struct_write_field_string(
    VortexWriter* writer,
    size_t column_index,
    size_t field_index,
    const uint8_t* data,
    const uint64_t* offsets,
    const uint8_t* null_map,
    size_t num_rows
);

/// Finalize the struct column data.
///
/// Must be called after all fields are written.
///
/// @param writer Writer handle.
/// @param column_index Column index of the struct column.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_struct_end(
    VortexWriter* writer,
    size_t column_index
);

// =============================================================================
// Writer Finalization
// =============================================================================

/// Finalize the writer and flush all data.
///
/// @param writer Writer handle.
/// @return 0 on success, negative error code on failure.
int32_t vortex_writer_finalize(VortexWriter* writer);

// =============================================================================
// Writer Information
// =============================================================================

/// Get the number of columns in the writer's schema.
///
/// @param writer Writer handle.
/// @return Number of columns, or 0 if writer is NULL.
size_t vortex_writer_num_columns(const VortexWriter* writer);

/// Get the total number of rows written.
///
/// @param writer Writer handle.
/// @return Total rows written, or 0 if writer is NULL.
size_t vortex_writer_total_rows(const VortexWriter* writer);

#ifdef __cplusplus
}
#endif
