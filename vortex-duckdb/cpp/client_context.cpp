// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"

#include "duckdb_vx/duckdb_diagnostics.h"
DUCKDB_INCLUDES_BEGIN
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/storage/object_cache.hpp>
DUCKDB_INCLUDES_END

extern "C" duckdb_client_context duckdb_vx_connection_get_client_context(duckdb_connection conn) {
    try {
        auto connection = reinterpret_cast<duckdb::Connection *>(conn);
        return reinterpret_cast<duckdb_client_context>(connection->context.get());
    } catch (...) {
        return nullptr;
    }
}

extern "C" duckdb_vx_object_cache duckdb_client_context_get_object_cache(duckdb_client_context ffi_context) {
    try {
        auto *context = reinterpret_cast<duckdb::ClientContext *>(ffi_context);
        // This is okay because this is a ref to the object cache, this lives as long as the database.
        return reinterpret_cast<duckdb_vx_object_cache>(&duckdb::ObjectCache::GetObjectCache(*context));
    } catch (...) {
        return nullptr;
    }
}

extern "C" duckdb_value duckdb_client_context_try_get_current_setting(duckdb_client_context context,
                                                                      const char *key) {
    if (!context || !key) {
        return nullptr;
    }

    try {
        auto client_context = reinterpret_cast<duckdb::ClientContext *>(context);
        duckdb::Value result;
        auto lookup_result = client_context->TryGetCurrentSetting(key, result);

        if (lookup_result) {
            return reinterpret_cast<duckdb_value>(new duckdb::Value(result));
        }

        return nullptr;
    } catch (...) {
        return nullptr;
    }
}
