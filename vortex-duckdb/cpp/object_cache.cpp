// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
#include "duckdb/storage/object_cache.hpp"

extern "C" void duckdb_vx_object_cache_put(duckdb_vx_object_cache cache, const char *key, void *value) {
    auto object_cache = reinterpret_cast<duckdb::ObjectCache *>(cache);
    object_cache->Put(std::string(key), value)
}

extern "C" void *duckdb_vx_object_cache_get(duckdb_vx_object_cache cache, const char *key) {
    auto object_cache = reinterpret_cast<duckdb::ObjectCache *>(cache);
    object_cache->Get<void *>(std::string(key));
}