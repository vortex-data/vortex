// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/object_cache.h"

DUCKDB_INCLUDES_BEGIN
#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/object_cache.hpp>
DUCKDB_INCLUDES_END

#include <string>
#include <utility>

namespace {

class VortexObjectCacheEntry final : public duckdb::ObjectCacheEntry {
public:
    VortexObjectCacheEntry(std::string object_type_p,
                           duckdb::idx_t estimated_memory_p,
                           void *data_p,
                           duckdb_delete_callback_t delete_callback_p)
        : object_type(std::move(object_type_p)), estimated_memory(estimated_memory_p), data(data_p),
          delete_callback(delete_callback_p) {
    }

    ~VortexObjectCacheEntry() override {
        if (delete_callback && data) {
            delete_callback(data);
        }
    }

    std::string GetObjectType() override {
        return object_type;
    }

    duckdb::optional_idx GetEstimatedCacheMemory() const override {
        return estimated_memory;
    }

    void *GetData() const {
        return data;
    }

private:
    std::string object_type;
    duckdb::idx_t estimated_memory;
    void *data;
    duckdb_delete_callback_t delete_callback;
};

} // namespace

struct duckdb_vx_object_cache_entry_ {
    explicit duckdb_vx_object_cache_entry_(duckdb::shared_ptr<duckdb::ObjectCacheEntry> entry_p)
        : entry(std::move(entry_p)) {
    }

    duckdb::shared_ptr<duckdb::ObjectCacheEntry> entry;
};

extern "C" duckdb_vx_object_cache_entry duckdb_vx_object_cache_get(duckdb_client_context ctx,
                                                                   const char *key,
                                                                   size_t key_len,
                                                                   const char *object_type) {
    if (!ctx || !key || !object_type) {
        return nullptr;
    }

    try {
        auto &context = *reinterpret_cast<duckdb::ClientContext *>(ctx);
        auto object = duckdb::ObjectCache::GetObjectCache(context).GetObject(std::string(key, key_len));
        if (!object || object->GetObjectType() != object_type) {
            return nullptr;
        }
        if (!dynamic_cast<VortexObjectCacheEntry *>(object.get())) {
            return nullptr;
        }
        return new duckdb_vx_object_cache_entry_(std::move(object));
    } catch (...) {
        return nullptr;
    }
}

extern "C" void *duckdb_vx_object_cache_entry_get_data(duckdb_vx_object_cache_entry entry) {
    if (!entry) {
        return nullptr;
    }

    auto *vortex_entry = dynamic_cast<VortexObjectCacheEntry *>(entry->entry.get());
    return vortex_entry ? vortex_entry->GetData() : nullptr;
}

extern "C" void duckdb_vx_object_cache_entry_free(duckdb_vx_object_cache_entry *entry) {
    if (!entry || !*entry) {
        return;
    }
    delete *entry;
    *entry = nullptr;
}

extern "C" duckdb_state duckdb_vx_object_cache_put(duckdb_client_context ctx,
                                                   const char *key,
                                                   size_t key_len,
                                                   const char *object_type,
                                                   idx_t estimated_memory,
                                                   void *data,
                                                   duckdb_delete_callback_t delete_callback) {
    bool entry_created = false;
    try {
        if (!ctx || !key || !object_type || !data) {
            if (delete_callback && data) {
                delete_callback(data);
            }
            return DuckDBError;
        }

        auto &context = *reinterpret_cast<duckdb::ClientContext *>(ctx);
        auto entry = duckdb::make_shared_ptr<VortexObjectCacheEntry>(
            object_type, estimated_memory, data, delete_callback);
        entry_created = true;
        duckdb::ObjectCache::GetObjectCache(context).Put(std::string(key, key_len), std::move(entry));
        return DuckDBSuccess;
    } catch (...) {
        if (!entry_created && delete_callback && data) {
            delete_callback(data);
        }
        return DuckDBError;
    }
}
