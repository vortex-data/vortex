// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "common.h"

// Exporter API for extracting data from Vortex arrays.
//
// Exporters are returned by vortex_scanner_read_batch() and provide a way to
// copy data from Vortex arrays into ClickHouse column buffers.
//
// Usage:
//   VortexExporterHandle* batch = vortex_scanner_read_batch(scanner);
//   while (vortex_exporter_has_more(batch)) {
//       int64_t rows = vortex_exporter_export(batch, buffer, buffer_size);
//       if (rows < 0) { /* handle error */ }
//       // Process exported rows...
//   }
//   vortex_exporter_free(batch);

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Exporter Lifecycle
// =============================================================================

/// Free an exporter handle.
///
/// @param handle Exporter handle to free. NULL is safely ignored.
void vortex_exporter_free(VortexExporterHandle* handle);

// =============================================================================
// Data Export
// =============================================================================

/// Check if the exporter has more data to export.
///
/// @param handle Exporter handle.
/// @return 1 if more data available, 0 if no more data or handle is NULL.
int32_t vortex_exporter_has_more(const VortexExporterHandle* handle);

/// Get the total number of rows in the exporter.
///
/// @param handle Exporter handle.
/// @return Total number of rows, or 0 if handle is NULL.
size_t vortex_exporter_len(const VortexExporterHandle* handle);

/// Export data to a buffer.
///
/// This function copies data from the Vortex array into the provided buffer.
/// The buffer must be pre-allocated by the caller with sufficient size.
///
/// For primitive types (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
/// Float32, Float64), the buffer should be an array of the corresponding C type.
///
/// For example, for Int64 columns:
///   int64_t* buffer = (int64_t*)malloc(sizeof(int64_t) * max_rows);
///   int64_t rows = vortex_exporter_export(handle, buffer, sizeof(int64_t) * max_rows, max_rows);
///
/// @param handle Exporter handle.
/// @param buffer Pointer to pre-allocated buffer.
/// @param buffer_size_bytes Size of the buffer in bytes.
/// @param max_rows Maximum number of rows to export.
/// @return Number of rows actually exported, or negative on error:
///         -1: handle or buffer is NULL
///         -2: export failed
int64_t vortex_exporter_export(
    VortexExporterHandle* handle,
    void* buffer,
    size_t buffer_size_bytes,
    size_t max_rows
);

// =============================================================================
// Struct Exporter (for multi-column data)
// =============================================================================

/// Get the number of fields in a struct exporter.
///
/// This function is only valid for struct-typed arrays.
///
/// @param handle Exporter handle.
/// @return Number of fields, or 0 if not a struct or handle is NULL.
size_t vortex_exporter_num_fields(const VortexExporterHandle* handle);

/// Get a field exporter from a struct exporter.
///
/// This function returns an exporter for a specific field of a struct.
/// The returned exporter is owned by the caller and must be freed.
///
/// @param handle Struct exporter handle.
/// @param index Field index (0-based).
/// @return Field exporter handle, or NULL on error.
VortexExporterHandle* vortex_exporter_get_field(
    VortexExporterHandle* handle,
    size_t index
);

// =============================================================================
// String/VarBinView Exporter
// =============================================================================

/// Export string data.
///
/// This function exports variable-length string data. For each row, it writes:
/// - The string length to lengths[i]
/// - The string data starting at data + offsets[i]
///
/// @param handle Exporter handle.
/// @param data Buffer for string data (concatenated).
/// @param lengths Buffer for string lengths.
/// @param offsets Buffer for string offsets in data buffer.
/// @param max_rows Maximum number of rows to export.
/// @return Number of rows actually exported, or negative on error.
int64_t vortex_exporter_export_strings(
    VortexExporterHandle* handle,
    char* data,
    uint32_t* lengths,
    uint64_t* offsets,
    size_t max_rows
);

/// Get the total size of string data for the remaining rows.
///
/// This function is useful for pre-allocating buffers before calling
/// vortex_exporter_export_strings. It returns the total number of bytes
/// needed for all string data and the number of remaining rows.
///
/// @param handle Exporter handle (must be a string exporter).
/// @param total_bytes Output parameter for total bytes needed.
/// @param num_rows Output parameter for number of remaining rows.
/// @return 0 on success, negative on error:
///         -1: handle or output pointers are NULL
///         -2: not a string exporter or other error
int32_t vortex_exporter_string_data_size(
    const VortexExporterHandle* handle,
    size_t* total_bytes,
    size_t* num_rows
);

// =============================================================================
// Nullable Data Support
// =============================================================================

/// Export validity (null) bitmap.
///
/// This function exports the validity bitmap for nullable columns.
/// Each bit indicates whether the corresponding row is valid (1) or null (0).
///
/// The bitmap is stored in little-endian byte order, with the first row
/// corresponding to the least significant bit of the first byte.
///
/// @param handle Exporter handle.
/// @param validity_bitmap Buffer for validity bitmap.
///        Size must be at least (max_rows + 7) / 8 bytes.
/// @param max_rows Maximum number of rows.
/// @return Number of rows, or negative on error.
int64_t vortex_exporter_export_validity(
    VortexExporterHandle* handle,
    uint8_t* validity_bitmap,
    size_t max_rows
);

/// Check if the column is nullable.
///
/// @param handle Exporter handle.
/// @return 1 if nullable, 0 if non-nullable or handle is NULL.
int32_t vortex_exporter_is_nullable(const VortexExporterHandle* handle);

// =============================================================================
// List/Array Exporter
// =============================================================================

/// Check if the exporter is a list (array) exporter.
///
/// List exporters are used for ClickHouse Array columns.
///
/// @param handle Exporter handle.
/// @return 1 if it's a list exporter, 0 otherwise.
int32_t vortex_exporter_is_list(const VortexExporterHandle* handle);

/// Export list offsets.
///
/// For list arrays (ClickHouse Array columns), this exports the offsets
/// that indicate where each array element starts in the flattened elements.
///
/// The offsets array will have num_rows + 1 elements written, where:
/// - offsets[i] is the start index of array i in the flattened elements
/// - offsets[num_rows] is the total number of elements
///
/// @param handle Exporter handle (must be a list exporter).
/// @param offsets Buffer for offsets (must have space for max_rows + 1 uint64_t values).
/// @param max_rows Maximum number of rows (arrays) to export.
/// @return Number of rows (arrays) exported, or negative on error.
int64_t vortex_exporter_export_list_offsets(
    VortexExporterHandle* handle,
    uint64_t* offsets,
    size_t max_rows
);

/// Get the element exporter from a list exporter.
///
/// This returns an exporter for the flattened elements of all arrays.
/// Use this exporter to export the actual element data after exporting offsets.
///
/// @param handle List exporter handle.
/// @return Element exporter handle, or NULL on error. Caller must free.
VortexExporterHandle* vortex_exporter_get_list_elements(
    VortexExporterHandle* handle
);

/// Get the total number of elements in all arrays (for a list exporter).
///
/// This is useful for pre-allocating the element buffer.
///
/// @param handle List exporter handle.
/// @return Total number of elements, or 0 if not a list exporter or on error.
size_t vortex_exporter_list_total_elements(const VortexExporterHandle* handle);

#ifdef __cplusplus
}
#endif
