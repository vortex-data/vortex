// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
DUCKDB_INCLUDES_END

#include "duckdb_vx/client_context.h"
#include "duckdb_vx/error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_file_handle_ *duckdb_vx_file_handle;

// Open a file using DuckDB's filesystem (supports httpfs, s3, etc.).
duckdb_vx_file_handle
duckdb_vx_fs_open(duckdb_client_context ctx, const char *path, duckdb_vx_error *error_out);

// Close a previously opened file handle.
void duckdb_vx_fs_close(duckdb_vx_file_handle *handle);

// Get the size of an opened file.
duckdb_state duckdb_vx_fs_get_size(duckdb_vx_file_handle handle, idx_t *size_out, duckdb_vx_error *error_out);

// Read up to len bytes at the given offset into buffer. Returns bytes read via out_len.
// TODO(myrrc) Here we use duckdb's positional read which returns nothing,
//  and thus we 1. don't know whether this read succeeded in the function itself,
//  only when out-of-bounds exception is thrown
//  2. Always have out_len=len.
//  Maybe the issue is in past-the-end reads which propagate an Error
//  in rust code, but here we throw an exception
duckdb_state duckdb_vx_fs_read(duckdb_vx_file_handle handle,
                               idx_t offset,
                               idx_t len,
                               uint8_t *buffer,
                               idx_t *out_len,
                               duckdb_vx_error *error_out);

/// Callback invoked for each entry returned by `duckdb_vx_fs_list_files`.
///
/// @param name  The entry's path, full for remote files, relative for local files
/// @param is_dir  Whether the entry is a directory.
/// @param user_data  Opaque pointer forwarded from the caller.
typedef void (*duckdb_vx_list_files_callback)(const char *name, bool is_dir, void *user_data);

/// Non-recursively list entries in a directory using DuckDB's filesystem.
///
/// Invokes `callback` once for each entry (file or subdirectory) found directly
/// inside `directory`.  The caller is responsible for recursing into subdirectories
/// if a recursive listing is desired.
duckdb_state duckdb_vx_fs_list_files(duckdb_client_context ctx,
                                     const char *directory,
                                     duckdb_vx_list_files_callback callback,
                                     void *user_data,
                                     duckdb_vx_error *error_out);

// Create/truncate a file for writing using DuckDB's filesystem.
duckdb_vx_file_handle
duckdb_vx_fs_create(duckdb_client_context ctx, const char *path, duckdb_vx_error *error_out);

// Write len bytes at the given offset from buffer.
duckdb_state duckdb_vx_fs_write(duckdb_vx_file_handle handle,
                                idx_t offset,
                                idx_t len,
                                uint8_t *buffer,
                                idx_t *out_len,
                                duckdb_vx_error *error_out);

// Flush pending writes to storage.
duckdb_state duckdb_vx_fs_sync(duckdb_vx_file_handle handle, duckdb_vx_error *error_out);

#ifdef __cplusplus /* End C ABI */
}
#endif
