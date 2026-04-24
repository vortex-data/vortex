// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "include/duckdb_vx/vector.h"
#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx.h"
#include "duckdb_vx/vector_buffer.hpp"

using namespace duckdb;

extern "C" void duckdb_vx_vector_slice_to_dictionary(duckdb_vector ffi_vector,
                                                     duckdb_selection_vector ffi_sel_vec,
                                                     idx_t selection_vector_length) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto sel_vec = reinterpret_cast<SelectionVector *>(ffi_sel_vec);
    vector->Slice(*sel_vec, selection_vector_length);
}

extern "C" void duckdb_vx_vector_dictionary(duckdb_vector ffi_vector,
                                            duckdb_vector ffi_dict,
                                            idx_t dictionary_size,
                                            duckdb_selection_vector ffi_sel_vec,
                                            idx_t count) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto dict = reinterpret_cast<Vector *>(ffi_dict);
    auto sel_vec = reinterpret_cast<SelectionVector *>(ffi_sel_vec);
    vector->Dictionary(*dict, dictionary_size, *sel_vec, count);
}

extern "C" void duckdb_vx_set_dictionary_vector_length(duckdb_vector dict, unsigned int len) {
    auto ddict = reinterpret_cast<duckdb::Vector *>(dict);
    ddict->GetBuffer()->Cast<DictionaryBuffer>().SetDictionarySize(len);
}

extern "C" void
duckdb_vx_sequence_vector(duckdb_vector c_vector, int64_t start, int64_t step, idx_t capacity) {
    auto vector = reinterpret_cast<Vector *>(c_vector);
    vector->Sequence(start, step, capacity);
}

namespace vortex {

// This is a complete hack to access the data buffer and pointer of a vector.
class DataVector : public Vector {
public:
    inline void SetDataBuffer(buffer_ptr<VectorBuffer> new_buffer) {
        buffer = std::move(new_buffer);
    };

    inline void SetDataPtr(data_ptr_t ptr) {
        data = ptr;
    };

    inline ValidityMask &GetValidity() {
        return validity;
    };
};

// Same hack for ValidityMask: access protected fields via inheritance.
class ExternalValidityMask : public ValidityMask {
public:
    inline void SetExternal(idx_t u64_offset, idx_t cap, buffer_ptr<ValidityBuffer> keeper) {
        validity_data = std::move(keeper);
        // Derive validity_mask from validity_data so the two stay consistent.
        validity_mask = reinterpret_cast<validity_t *>(validity_data.get()) + u64_offset;
        capacity = cap;
    };
};

} // namespace vortex

extern "C" void duckdb_vx_string_vector_add_vector_data_buffer(duckdb_vector ffi_vector,
                                                               duckdb_vx_vector_buffer buffer) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto data = reinterpret_cast<shared_ptr<vortex::ExternalVectorBuffer> *>(buffer);
    StringVector::AddBuffer(*vector, *data);
}

extern "C" void duckdb_vx_vector_set_vector_data_buffer(duckdb_vector ffi_vector,
                                                        duckdb_vx_vector_buffer buffer) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto dvector = reinterpret_cast<vortex::DataVector *>(vector);
    auto data = reinterpret_cast<shared_ptr<vortex::ExternalVectorBuffer> *>(buffer);
    dvector->SetDataBuffer(*data);
}

extern "C" void duckdb_vx_vector_set_data_ptr(duckdb_vector ffi_vector, void *ptr) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto dvector = reinterpret_cast<vortex::DataVector *>(vector);
    dvector->SetDataPtr((data_ptr_t)ptr);
}

extern "C" void duckdb_vx_vector_set_validity_data(duckdb_vector ffi_vector,
                                                   idx_t u64_offset,
                                                   idx_t capacity,
                                                   duckdb_vx_vector_buffer buffer,
                                                   void *data_ptr) {
    auto dvector = reinterpret_cast<vortex::DataVector *>(ffi_vector);
    auto &validity = dvector->GetValidity();
    // ExternalValidityMask adds no members, so this downcast only exposes
    // access to ValidityMask's protected fields.
    auto ext_validity = static_cast<vortex::ExternalValidityMask *>(&validity);

    // Use the shared_ptr aliasing constructor: the control block ref-counts the
    // ExternalVectorBuffer (preventing the Rust buffer from being freed),
    // while the stored pointer points to the explicit data_ptr.
    auto ext_buf = reinterpret_cast<shared_ptr<vortex::ExternalVectorBuffer> *>(buffer);
    auto keeper = shared_ptr<TemplatedValidityData<validity_t>>(
        *ext_buf,
        reinterpret_cast<TemplatedValidityData<validity_t> *>(data_ptr));

    // Set validity_data, derive validity_mask from it at u64_offset, and set capacity.
    ext_validity->SetExternal(u64_offset, capacity, std::move(keeper));
}

extern "C" duckdb_value duckdb_vx_vector_get_value(duckdb_vector ffi_vector, idx_t index) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto value = duckdb::make_uniq<Value>(vector->GetValue(index));
    return reinterpret_cast<duckdb_value>(value.release());
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
        *err = nullptr;
        return result;
    } catch (std::runtime_error &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
        return nullptr;
    }
}

void duckdb_vx_vector_set_all_valid(duckdb_vector ffi_vector) {
    using enum VectorType;
    Vector &vector = *reinterpret_cast<Vector *>(ffi_vector);
    const VectorType type = vector.GetVectorType();
    D_ASSERT(type != DICTIONARY_VECTOR && type != SEQUENCE_VECTOR);
    switch (type) {
    case CONSTANT_VECTOR:
        return ConstantVector::Validity(vector).Reset();
    case FLAT_VECTOR:
        return FlatVector::Validity(vector).Reset();
    case FSST_VECTOR:
        return FSSTVector::Validity(vector).Reset();
    default:
        __builtin_unreachable();
    }
}
