// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/types/vector.hpp"
#include "duckdb_vx.h"

using namespace duckdb;

// buffer_ptr is shared_ptr, two pointers long, but duckdb_vx_reusable_dict is
// one pointer long, so we need a wrapper.
using Buffer = buffer_ptr<VectorChildBuffer>;
struct ReusableDict {
    Buffer buffer;
    ReusableDict(Buffer buffer) : buffer(std::move(buffer)) {
    }
};

extern "C" duckdb_vx_reusable_dict duckdb_vx_reusable_dict_create(duckdb_logical_type ffi_type, idx_t size) {
    const LogicalType &type = *reinterpret_cast<LogicalType *>(ffi_type);
    auto buffer = DictionaryVector::CreateReusableDictionary(type, size);
    auto ptr = std::make_unique<ReusableDict>(std::move(buffer));
    return reinterpret_cast<duckdb_vx_reusable_dict>(ptr.release());
}

extern "C" void duckdb_vx_reusable_dict_destroy(duckdb_vx_reusable_dict *dict) {
    if (dict && *dict) {
        delete reinterpret_cast<ReusableDict *>(*dict);
    }
}

extern "C" duckdb_vx_reusable_dict duckdb_vx_reusable_dict_clone(duckdb_vx_reusable_dict dict) {
    ReusableDict *wrapper = reinterpret_cast<ReusableDict *>(dict);
    auto ptr = std::make_unique<ReusableDict>(wrapper->buffer);
    return reinterpret_cast<duckdb_vx_reusable_dict>(ptr.release());
}

extern "C" void duckdb_vx_reusable_dict_set_vector(duckdb_vx_reusable_dict reusable,
                                                   duckdb_vector *out_vector) {
    auto *wrapper = reinterpret_cast<ReusableDict *>(reusable);
    *out_vector = reinterpret_cast<duckdb_vector>(&wrapper->buffer->data);
}

extern "C" void duckdb_vx_vector_dictionary_reusable(duckdb_vector ffi_vector,
                                                     duckdb_vx_reusable_dict reusable,
                                                     duckdb_selection_vector ffi_sel_vec) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto *wrapper = reinterpret_cast<ReusableDict *>(reusable);
    auto sel_vec = reinterpret_cast<SelectionVector *>(ffi_sel_vec);
    vector->Dictionary(wrapper->buffer, *sel_vec);
}
