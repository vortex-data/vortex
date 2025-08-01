// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"

#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/object_cache.hpp>

namespace vortex {
extern "C" duckdb_vx_object_cache
duckdb_vx_client_context_get_object_cache(duckdb_vx_client_context context) {
    auto client_context = reinterpret_cast<duckdb::ClientContext *>(context);
    return reinterpret_cast<duckdb_vx_object_cache>(&duckdb::ObjectCache::GetObjectCache(*client_context));
}
} // namespace vortex