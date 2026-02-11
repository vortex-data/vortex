// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cstdint>
#include <cstddef>

// Common types and definitions for vortex-clickhouse.

#ifdef __cplusplus
extern "C" {
#endif

/// Result status for operations.
typedef enum {
    VORTEX_CH_SUCCESS = 0,
    VORTEX_CH_ERROR = 1,
} vortex_ch_status;

/// Opaque handle to Vortex scanner.
/// Created by `vortex_scanner_new`, freed by `vortex_scanner_free`.
///
/// Thread safety: NOT thread-safe. All calls on a given handle must be
/// serialized by the caller. See scanner.h for details.
typedef struct VortexScanner VortexScanner;

/// Opaque handle to Vortex writer.
/// Created by vortex_writer_new(), freed by vortex_writer_free().
typedef struct VortexWriter VortexWriter;

/// Opaque handle to a column exporter.
/// Created by vortex_scanner_read_batch(), freed by vortex_exporter_free().
typedef struct VortexExporterHandle VortexExporterHandle;

// =============================================================================
// Error Handling API
// =============================================================================

/// Get the last error message.
///
/// Returns a null-terminated C string with the last error message,
/// or NULL if no error was set. The returned string must be freed
/// by calling `vortex_free_string()`.
///
/// @return Error message string, or NULL if no error. Caller must free.
char* vortex_get_last_error(void);

/// Check if there is a pending error.
///
/// @return 1 if an error is set, 0 otherwise.
int32_t vortex_has_error(void);

/// Clear the last error.
///
/// Call this before starting a new operation if you want to ensure
/// no stale error messages are present.
void vortex_clear_error(void);

/// Free a string returned by vortex FFI functions.
///
/// This function must be called to free strings returned by functions like
/// `vortex_get_last_error()`, `vortex_scanner_column_name()`, etc.
///
/// @param ptr String pointer to free. NULL is safely ignored.
void vortex_free_string(char* ptr);

#ifdef __cplusplus
}
#endif
