// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/types/vector.hpp"

#include "duckdb_vx.h"

using namespace duckdb;

namespace vortex {
// Wrapper struct to hold the buffer_ptr<VectorChildBuffer>
struct ReusableDict {
    buffer_ptr<VectorChildBuffer> buffer;

    explicit ReusableDict(buffer_ptr<VectorChildBuffer> buf) : buffer(std::move(buf)) {
    }
};
} // namespace vortex

extern "C" duckdb_vx_reusable_dict duckdb_vx_reusable_dict_create(duckdb_logical_type ffi_type, idx_t size) {
    auto type = reinterpret_cast<duckdb::LogicalType *>(ffi_type);
    auto buffer = DictionaryVector::CreateReusableDictionary(*type, size);
    auto *wrapper = new vortex::ReusableDict(std::move(buffer));
    return reinterpret_cast<duckdb_vx_reusable_dict>(wrapper);
}

extern "C" void duckdb_vx_reusable_dict_destroy(duckdb_vx_reusable_dict *dict) {
    if (dict != nullptr && *dict != nullptr) {
        auto *wrapper = reinterpret_cast<vortex::ReusableDict *>(*dict);
        delete wrapper;
        *dict = nullptr;
    }
}

extern "C" duckdb_vx_reusable_dict duckdb_vx_reusable_dict_clone(duckdb_vx_reusable_dict dict) {
    auto *wrapper = reinterpret_cast<vortex::ReusableDict *>(dict);
    auto *cloned = new vortex::ReusableDict(wrapper->buffer);
    return reinterpret_cast<duckdb_vx_reusable_dict>(cloned);
}

extern "C" void duckdb_vx_reusable_dict_set_vector(duckdb_vx_reusable_dict reusable,
                                                   duckdb_vector *out_vector) {
    auto *wrapper = reinterpret_cast<vortex::ReusableDict *>(reusable);
    *out_vector = reinterpret_cast<duckdb_vector>(&wrapper->buffer->data);
}

extern "C" void duckdb_vx_vector_dictionary_reusable(duckdb_vector ffi_vector,
                                                     duckdb_vx_reusable_dict reusable,
                                                     duckdb_selection_vector ffi_sel_vec) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto *wrapper = reinterpret_cast<vortex::ReusableDict *>(reusable);
    auto sel_vec = reinterpret_cast<SelectionVector *>(ffi_sel_vec);
    vector->Dictionary(wrapper->buffer, *sel_vec);
}
