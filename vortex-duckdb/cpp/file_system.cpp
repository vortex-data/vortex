// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
#include "duckdb_vx/error.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/helper.hpp>
#include <duckdb/main/client_context.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <utility>

using namespace duckdb;

struct FileHandleWrapper {
    explicit FileHandleWrapper(unique_ptr<FileHandle> handle_p) : handle(std::move(handle_p)) {
    }

    unique_ptr<FileHandle> handle;
};

using vortex::HandleException;
using vortex::SetError;

extern "C" duckdb_vx_file_handle duckdb_vx_fs_open(duckdb_vx_client_context ctx, const char *path,
                                                     duckdb_vx_error *error_out) {
    if (!ctx || !path) {
        SetError(error_out, "Invalid filesystem open arguments");
        return nullptr;
    }

    try {
        auto *client_context = reinterpret_cast<ClientContext *>(ctx);
        auto &fs = FileSystem::GetFileSystem(*client_context);
        auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_PARALLEL_ACCESS);
        return reinterpret_cast<duckdb_vx_file_handle>(new FileHandleWrapper(std::move(handle)));
    } catch (...) {
        HandleException(std::current_exception(), error_out);
        return nullptr;
    }
}

extern "C" duckdb_vx_file_handle duckdb_vx_fs_create(duckdb_vx_client_context ctx, const char *path,
                                                      duckdb_vx_error *error_out) {
    if (!ctx || !path) {
        SetError(error_out, "Invalid filesystem create arguments");
        return nullptr;
    }

    try {
        auto *client_context = reinterpret_cast<ClientContext *>(ctx);
        auto &fs = FileSystem::GetFileSystem(*client_context);
        auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE |
                                            FileFlags::FILE_FLAGS_PARALLEL_ACCESS);
        handle->Truncate(0);
        return reinterpret_cast<duckdb_vx_file_handle>(new FileHandleWrapper(std::move(handle)));
    } catch (...) {
        HandleException(std::current_exception(), error_out);
        return nullptr;
    }
}

extern "C" void duckdb_vx_fs_close(duckdb_vx_file_handle *handle) {
    if (!handle || !*handle) {
        return;
    }
    auto wrapper = reinterpret_cast<FileHandleWrapper *>(*handle);
    delete wrapper;
    *handle = nullptr;
}

extern "C" duckdb_state duckdb_vx_fs_get_size(duckdb_vx_file_handle handle, idx_t *size_out,
                                               duckdb_vx_error *error_out) {
    if (!handle || !size_out) {
        SetError(error_out, "Invalid arguments to fs_get_size");
        return DuckDBError;
    }

    try {
        auto *wrapper = reinterpret_cast<FileHandleWrapper *>(handle);
        *size_out = wrapper->handle->GetFileSize();
        return DuckDBSuccess;
    } catch (...) {
        return HandleException(std::current_exception(), error_out);
    }
}

extern "C" duckdb_state duckdb_vx_fs_read(duckdb_vx_file_handle handle, idx_t offset, idx_t len, uint8_t *buffer,
                                            idx_t *out_len, duckdb_vx_error *error_out) {
    if (!handle || !buffer || !out_len) {
        SetError(error_out, "Invalid arguments to fs_read");
        return DuckDBError;
    }

    try {
        auto *wrapper = reinterpret_cast<FileHandleWrapper *>(handle);
        wrapper->handle->Read(buffer, len, offset);
        *out_len = len;
        return DuckDBSuccess;
    } catch (...) {
        return HandleException(std::current_exception(), error_out);
    }
}

extern "C" duckdb_state duckdb_vx_fs_write(duckdb_vx_file_handle handle, idx_t offset, idx_t len,
                                            uint8_t *buffer, idx_t *out_len,
                                            duckdb_vx_error *error_out) {
    if (!handle || !buffer || !out_len) {
        SetError(error_out, "Invalid arguments to fs_write");
        return DuckDBError;
    }

    try {
        auto *wrapper = reinterpret_cast<FileHandleWrapper *>(handle);
        wrapper->handle->Write(QueryContext(), buffer, len, offset);
        *out_len = len;
        return DuckDBSuccess;
    } catch (...) {
        return HandleException(std::current_exception(), error_out);
    }
}

extern "C" duckdb_vx_uri_list duckdb_vx_fs_glob(duckdb_vx_client_context ctx, const char *pattern,
                                                duckdb_vx_error *error_out) {
    duckdb_vx_uri_list result{nullptr, 0};

    if (!ctx || !pattern) {
        SetError(error_out, "Invalid arguments to fs_glob");
        return result;
    }

    try {
        auto *client_context = reinterpret_cast<ClientContext *>(ctx);
        auto &fs = FileSystem::GetFileSystem(*client_context);
        auto matches = fs.Glob(pattern);

        if (matches.empty()) {
            return result;
        }

        result.count = matches.size();
        result.entries = static_cast<const char **>(duckdb_malloc(sizeof(char *) * matches.size()));
        for (size_t i = 0; i < matches.size(); i++) {
            const auto &entry = matches[i].path;
            auto *owned = static_cast<char *>(duckdb_malloc(entry.size() + 1));
            std::memcpy(owned, entry.data(), entry.size());
            owned[entry.size()] = '\0';
            result.entries[i] = owned;
        }

        return result;
    } catch (...) {
        HandleException(std::current_exception(), error_out);
        return result;
    }
}

extern "C" void duckdb_vx_uri_list_free(duckdb_vx_uri_list *list) {
    if (!list || !list->entries) {
        return;
    }
    for (size_t i = 0; i < list->count; i++) {
        duckdb_free(const_cast<char *>(list->entries[i]));
    }
    duckdb_free(list->entries);
    list->entries = nullptr;
    list->count = 0;
}

extern "C" duckdb_state duckdb_vx_fs_sync(duckdb_vx_file_handle handle, duckdb_vx_error *error_out) {
    if (!handle) {
        SetError(error_out, "Invalid arguments to fs_sync");
        return DuckDBError;
    }

    try {
        auto *wrapper = reinterpret_cast<FileHandleWrapper *>(handle);
        wrapper->handle->Sync();
        return DuckDBSuccess;
    } catch (...) {
        return HandleException(std::current_exception(), error_out);
    }
}
