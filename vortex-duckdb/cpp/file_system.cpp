// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
#include "duckdb_vx/error.hpp"

#include "duckdb_vx/duckdb_diagnostics.h"
DUCKDB_INCLUDES_BEGIN
#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/helper.hpp>
#include <duckdb/main/client_context.hpp>
DUCKDB_INCLUDES_END

#include <utility>

using namespace duckdb;
using vortex::SetError;

extern "C" duckdb_vx_file_handle
duckdb_vx_fs_open(duckdb_client_context ctx, const char *path, duckdb_vx_error *error_out) {
    if (!ctx || !path) {
        SetError(error_out, "Invalid filesystem open arguments");
        return nullptr;
    }

    auto *client_context = reinterpret_cast<ClientContext *>(ctx);

    try {
        auto &fs = FileSystem::GetFileSystem(*client_context);
        auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_PARALLEL_ACCESS);
        return reinterpret_cast<duckdb_vx_file_handle>(handle.release());
    } catch (const std::exception &e) {
        SetError(error_out, e.what());
        return nullptr;
    }
}

extern "C" duckdb_vx_file_handle
duckdb_vx_fs_create(duckdb_client_context ctx, const char *path, duckdb_vx_error *error_out) {
    if (!ctx || !path) {
        SetError(error_out, "Invalid filesystem create arguments");
        return nullptr;
    }

    constexpr auto flags = FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW |
                           FileFlags::FILE_FLAGS_PARALLEL_ACCESS;
    auto *client_context = reinterpret_cast<ClientContext *>(ctx);

    try {
        auto &fs = FileSystem::GetFileSystem(*client_context);
        auto handle = fs.OpenFile(path, flags);
        return reinterpret_cast<duckdb_vx_file_handle>(handle.release());
    } catch (const std::exception &e) {
        SetError(error_out, e.what());
        return nullptr;
    }
}

extern "C" void duckdb_vx_fs_close(duckdb_vx_file_handle *handle) {
    if (handle && *handle) {
        delete reinterpret_cast<FileHandle *>(std::exchange(*handle, nullptr));
    }
}

extern "C" duckdb_state
duckdb_vx_fs_get_size(duckdb_vx_file_handle handle, idx_t *size_out, duckdb_vx_error *error_out) {
    if (!handle || !size_out) {
        return SetError(error_out, "Invalid arguments to fs_get_size");
    }

    try {
        *size_out = reinterpret_cast<FileHandle *>(handle)->GetFileSize();
    } catch (const std::exception &e) {
        return SetError(error_out, e.what());
    }
    return DuckDBSuccess;
}

extern "C" duckdb_state duckdb_vx_fs_read(duckdb_vx_file_handle handle,
                                          idx_t offset,
                                          idx_t len,
                                          uint8_t *buffer,
                                          idx_t *out_len,
                                          duckdb_vx_error *error_out) {
    if (!handle || !buffer || !out_len) {
        return SetError(error_out, "Invalid arguments to fs_read");
    }

    try {
        reinterpret_cast<FileHandle *>(handle)->Read(buffer, len, offset);
        *out_len = len;
    } catch (const std::exception &e) {
        return SetError(error_out, e.what());
    }
    return DuckDBSuccess;
}

extern "C" duckdb_state duckdb_vx_fs_write(duckdb_vx_file_handle handle,
                                           idx_t offset,
                                           idx_t len,
                                           uint8_t *buffer,
                                           idx_t *out_len,
                                           duckdb_vx_error *error_out) {
    if (!handle || !buffer || !out_len) {
        return SetError(error_out, "Invalid arguments to fs_write");
    }

    try {
        reinterpret_cast<FileHandle *>(handle)->Write(QueryContext(), buffer, len, offset);
        *out_len = len;
    } catch (const std::exception &e) {
        return SetError(error_out, e.what());
    }
    return DuckDBSuccess;
}

extern "C" duckdb_state duckdb_vx_fs_list_files(duckdb_client_context ctx,
                                                const char *directory,
                                                duckdb_vx_list_files_callback callback,
                                                void *user_data,
                                                duckdb_vx_error *error_out) {
    if (!ctx || !directory || !callback) {
        return SetError(error_out, "Invalid arguments to fs_list_files");
    }

    auto fn = [&](const string &name, bool is_dir) {
        callback(name.c_str(), is_dir, user_data);
    };
    auto *client_context = reinterpret_cast<ClientContext *>(ctx);

    try {
        FileSystem::GetFileSystem(*client_context).ListFiles(directory, fn);
        return DuckDBSuccess;
    } catch (const std::exception &e) {
        return SetError(error_out, e.what());
    }
    return DuckDBSuccess;
}

extern "C" duckdb_state duckdb_vx_fs_sync(duckdb_vx_file_handle handle, duckdb_vx_error *error_out) {
    if (!handle) {
        return SetError(error_out, "Invalid arguments to fs_sync");
    }

    try {
        reinterpret_cast<FileHandle *>(handle)->Sync();
    } catch (const std::exception &e) {
        return SetError(error_out, e.what());
    }
    return DuckDBSuccess;
}
