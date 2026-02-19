// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
#include "duckdb/storage/object_cache.hpp"

#include <iostream>

namespace vortex {

// Wrapper class to hold opaque pointers in DuckDB's object cache
class OpaqueWrapper : public duckdb::ObjectCacheEntry {
public:
    duckdb::unique_ptr<void, duckdb_vx_deleter_fn> ptr;

    explicit OpaqueWrapper(void *p, duckdb_vx_deleter_fn del) : ptr(p, del) {
    }
    ~OpaqueWrapper() override = default;

    duckdb::string GetObjectType() override {
        return "vortex_opaque_wrapper";
    }

    // Static method required by DuckDB's object cache
    static duckdb::string ObjectType() {
        return "vortex_opaque_wrapper";
    }
};

} // namespace vortex

extern "C" void duckdb_vx_object_cache_put(duckdb_vx_object_cache cache,
                                           const char *key,
                                           void *value,
                                           duckdb_vx_deleter_fn deleter) {
    auto object_cache = reinterpret_cast<duckdb::ObjectCache *>(cache);
    auto wrapper = duckdb::make_shared_ptr<vortex::OpaqueWrapper>(value, deleter);
    object_cache->Put(std::string(key), wrapper);
}

extern "C" void *duckdb_vx_object_cache_get(duckdb_vx_object_cache cache, const char *key) {
    auto object_cache = reinterpret_cast<duckdb::ObjectCache *>(cache);
    auto entry = object_cache->Get<vortex::OpaqueWrapper>(std::string(key));
    if (!entry) {
        return nullptr;
    }
    return entry->ptr.get();
}
