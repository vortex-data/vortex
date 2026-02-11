// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "common.h"

// Scanner API for reading Vortex files from ClickHouse.
//
// Thread safety:
//   A VortexScanner instance is NOT thread-safe. The caller must serialize all
//   calls to a given scanner handle. Do not call any scanner function from one
//   thread while another thread is using the same handle. Each thread should
//   either create its own scanner via `vortex_scanner_new`, or the caller must
//   use external synchronization (e.g., a mutex) to protect shared access.
//   The error reporting functions (`vortex_get_last_error`, `vortex_has_error`,
//   `vortex_clear_error`) use thread-local storage and are safe to call from
//   any thread.
//
// Usage:
//   VortexScanner* scanner = vortex_scanner_new("/path/to/file.vortex");
//   if (scanner == NULL) { /* handle error */ }
//
//   // Get schema information
//   size_t num_cols = vortex_scanner_num_columns(scanner);
//   for (size_t i = 0; i < num_cols; i++) {
//       const char* name = vortex_scanner_column_name(scanner, i);
//       const char* type = vortex_scanner_column_type(scanner, i);
//   }
//
//   // Set projection if needed
//   size_t indices[] = {0, 2};
//   vortex_scanner_set_projection(scanner, indices, 2);
//
//   // Read batches
//   while (vortex_scanner_has_more(scanner)) {
//       VortexExporterHandle* batch = vortex_scanner_read_batch(scanner);
//       // Export data from batch...
//       vortex_exporter_free(batch);
//   }
//
//   vortex_scanner_free(scanner);

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Scanner Creation and Destruction
// =============================================================================

/// Create a new Vortex scanner for the given path.
///
/// The path can be:
/// - A local file path: "/path/to/file.vortex"
/// - A local glob pattern: "/path/to/*.vortex"
/// - A remote URL: "s3://bucket/path/to/file.vortex"
/// - A remote glob: "s3://bucket/path/to/*.vortex"
///
/// @param path Null-terminated C string with the file path or glob pattern.
/// @return Scanner handle, or NULL on error.
VortexScanner* vortex_scanner_new(const char* path);

/// Free a Vortex scanner.
///
/// @param scanner Scanner handle to free. NULL is safely ignored.
void vortex_scanner_free(VortexScanner* scanner);

// =============================================================================
// Schema Introspection
// =============================================================================

/// Get the number of columns in the schema.
///
/// @param scanner Scanner handle.
/// @return Number of columns, or 0 if scanner is NULL.
size_t vortex_scanner_num_columns(const VortexScanner* scanner);

/// Get a column name by index.
///
/// @param scanner Scanner handle.
/// @param index Column index (0-based).
/// @return Null-terminated column name, or NULL if index is out of bounds.
///         The returned string is owned by the caller and must be freed
///         with vortex_free_string().
char* vortex_scanner_column_name(const VortexScanner* scanner, size_t index);

/// Get the ClickHouse type string for a column.
///
/// @param scanner Scanner handle.
/// @param index Column index (0-based).
/// @return Null-terminated ClickHouse type string (e.g., "Int64", "String"),
///         or NULL if index is out of bounds.
///         The returned string is owned by the caller and must be freed
///         with vortex_free_string().
char* vortex_scanner_column_type(const VortexScanner* scanner, size_t index);

// =============================================================================
// Scan Configuration
// =============================================================================

/// Set the columns to project (by index).
///
/// @param scanner Scanner handle.
/// @param indices Array of column indices to project.
/// @param num_indices Number of indices in the array.
/// @return 0 on success, negative error code on failure:
///         -1: scanner is NULL
///         -2: indices is NULL but num_indices > 0
///         -3: invalid column index
int32_t vortex_scanner_set_projection(
    VortexScanner* scanner,
    const size_t* indices,
    size_t num_indices
);

/// Set the batch size for reading.
///
/// @param scanner Scanner handle.
/// @param batch_size Number of rows per batch. Minimum is 1.
void vortex_scanner_set_batch_size(VortexScanner* scanner, size_t batch_size);

// =============================================================================
// Data Reading
// =============================================================================

/// Check if there are more batches to read.
///
/// @param scanner Scanner handle.
/// @return 1 if more data available, 0 if no more data or scanner is NULL.
int32_t vortex_scanner_has_more(const VortexScanner* scanner);

/// Read the next batch of data.
///
/// @param scanner Scanner handle.
/// @return Exporter handle for the batch, or NULL if no more data or on error.
///         The returned handle must be freed with vortex_exporter_free().
VortexExporterHandle* vortex_scanner_read_batch(VortexScanner* scanner);

// =============================================================================
// Progress Tracking
// =============================================================================

/// Get the number of files to scan.
///
/// @param scanner Scanner handle.
/// @return Number of files, or 0 if scanner is NULL.
size_t vortex_scanner_num_files(const VortexScanner* scanner);

/// Get the current file index being scanned.
///
/// @param scanner Scanner handle.
/// @return Current file index (0-based), or 0 if scanner is NULL.
size_t vortex_scanner_current_file_index(const VortexScanner* scanner);

/// Get the total number of rows read so far.
///
/// @param scanner Scanner handle.
/// @return Total rows read, or 0 if scanner is NULL.
uint64_t vortex_scanner_total_rows_read(const VortexScanner* scanner);

/// Get the total row count across all files.
///
/// This function reads metadata from all files to compute the total row count.
/// Note: This may be slow for large file sets as it opens each file's metadata.
///
/// @param scanner Scanner handle.
/// @return Total row count, or 0 on error. Call vortex_get_last_error() for details.
uint64_t vortex_scanner_total_row_count(const VortexScanner* scanner);

#ifdef __cplusplus
}
#endif
