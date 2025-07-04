// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/vector.hpp"

#include "duckdb_vx.h"
#include "duckdb_vx/data.hpp"

using namespace duckdb;

extern "C" void duckdb_vx_vector_slice_to_dictionary(duckdb_vector ffi_vector,
                                                     duckdb_selection_vector ffi_sel_vec,
                                                     idx_t selection_vector_length) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto sel_vec = reinterpret_cast<SelectionVector *>(ffi_sel_vec);
    vector->Slice(*sel_vec, selection_vector_length);
}

extern "C" void duckdb_vx_sequence_vector(duckdb_vector c_vector, int64_t start, int64_t step,
                                          idx_t capacity) {
    auto vector = reinterpret_cast<Vector *>(c_vector);
    vector->Sequence(start, step, capacity);
}

namespace vortex {

// This is a wrapper around an externally managed buffer, which can be assigned to a Vector and
// freed once the vector is done with the buffer.
class ExternalVectorBuffer : public VectorBuffer {
public:
    explicit ExternalVectorBuffer(unique_ptr<vortex::CData> data) : data(std::move(data)) {
    }

private:
    unique_ptr<vortex::CData> data;
};

} // namespace vortex

extern "C" void duckdb_vx_string_vector_add_buffer(duckdb_vector ffi_vector, duckdb_vx_data buffer) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto data = reinterpret_cast<vortex::CData *>(buffer);
    auto ext_buffer = duckdb::make_shared_ptr<vortex::ExternalVectorBuffer>(unique_ptr<vortex::CData>(data));
    StringVector::AddBuffer(*vector, ext_buffer);
}

void duckdb_vector_flatten(duckdb_vector vector, unsigned long len) {
    auto dvector = reinterpret_cast<Vector *>(vector);
    dvector->Flatten(len);
}

const char *duckdb_vector_to_string(duckdb_vector vector, unsigned long len, duckdb_vx_error *err) {
    try {
        auto dvector = reinterpret_cast<Vector *>(vector);
        auto str = dvector->ToString(len);
        auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
        memcpy(result, str.c_str(), str.size() + 1);
        err = nullptr;
        return result;
    } catch (std::runtime_error &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
        return nullptr;
    }
}