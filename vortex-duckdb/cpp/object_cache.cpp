// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
#include "duckdb/storage/object_cache.hpp"

#include <iostream>

namespace vortex {

// Function pointer type for custom deleter
typedef void (*deleter_fn_t)(void *ptr);

// Wrapper class to hold opaque pointers in DuckDB's object cache
class OpaqueWrapper : public duckdb::ObjectCacheEntry {
public:
    void *ptr;
    deleter_fn_t deleter;

    explicit OpaqueWrapper(void *p, deleter_fn_t del = nullptr) : ptr(p), deleter(del) {
    }

    ~OpaqueWrapper() override {
        if (deleter && ptr) {
            // Call the custom deleter function
            deleter(ptr);
        }
    }

    duckdb::string GetObjectType() override {
        return "vortex_opaque_wrapper";
    }

    // Static method required by DuckDB's object cache
    static duckdb::string ObjectType() {
        return "vortex_opaque_wrapper";
    }
};

} // namespace vortex

extern "C" void duckdb_vx_object_cache_put(duckdb_vx_object_cache cache, const char *key, void *value,
                                           vortex::deleter_fn_t deleter) {
    try {
        auto object_cache = reinterpret_cast<duckdb::ObjectCache *>(cache);
        auto wrapper = duckdb::make_shared_ptr<vortex::OpaqueWrapper>(value, deleter);
        object_cache->Put(std::string(key), wrapper);
    } catch (...) {
        // Silently fail on errors - could add error reporting later
    }
}

extern "C" void *duckdb_vx_object_cache_get(duckdb_vx_object_cache cache, const char *key) {
    try {
        auto object_cache = reinterpret_cast<duckdb::ObjectCache *>(cache);
        auto entry = object_cache->Get<vortex::OpaqueWrapper>(std::string(key));
        if (!entry) {
            return nullptr;
        }
        return entry->ptr;
    } catch (...) {
        return nullptr;
    }
}