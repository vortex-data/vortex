// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vector.h"
#include "vector.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

using namespace duckdb;

extern "C" duckdb_vector duckdb_vx_vector_slice(duckdb_vector ffi_vector, idx_t offset, idx_t end) {
    const Vector &vector = *reinterpret_cast<Vector *>(ffi_vector);
    Vector *const sliced = new Vector(vector, offset, end);
    return reinterpret_cast<duckdb_vector>(sliced);
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

// This is a complete hack to access the data buffer and pointer of a vector.
// Duckdb passes us Vectors and not VortexVectors. This only works because
// VortexVector doesn't add any members.
class VortexVector final : public Vector {
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
static_assert(sizeof(VortexVector) == sizeof(Vector));

// Same hack for ValidityMask: access protected fields via inheritance.
class ExternalValidityMask final : public ValidityMask {
public:
    inline void SetExternal(idx_t u64_offset, idx_t cap, buffer_ptr<ValidityBuffer> keeper) {
        validity_data = std::move(keeper);
        // Derive validity_mask from validity_data so the two stay consistent.
        validity_mask = reinterpret_cast<validity_t *>(validity_data.get()) + u64_offset;
        capacity = cap;
    };
};
static_assert(sizeof(ExternalValidityMask) == sizeof(ValidityMask));

extern "C" void duckdb_vx_string_vector_add_vector_data_buffer(duckdb_vector ffi_vector,
                                                               duckdb_vx_vector_buffer buffer) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto data = reinterpret_cast<shared_ptr<ExternalVectorBuffer> *>(buffer);
    StringVector::AddBuffer(*vector, *data);
}

extern "C" void duckdb_vx_vector_set_vector_data_buffer(duckdb_vector ffi_vector,
                                                        duckdb_vx_vector_buffer buffer) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto dvector = reinterpret_cast<VortexVector *>(vector);
    auto data = reinterpret_cast<shared_ptr<ExternalVectorBuffer> *>(buffer);
    dvector->SetDataBuffer(*data);
}

extern "C" void duckdb_vx_vector_set_data_ptr(duckdb_vector ffi_vector, void *ptr) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto dvector = reinterpret_cast<VortexVector *>(vector);
    dvector->SetDataPtr((data_ptr_t)ptr);
}

extern "C" void duckdb_vx_vector_set_validity_data(duckdb_vector ffi_vector,
                                                   idx_t u64_offset,
                                                   idx_t capacity,
                                                   duckdb_vx_vector_buffer buffer,
                                                   void *data_ptr) {
    auto dvector = reinterpret_cast<VortexVector *>(ffi_vector);
    auto &validity = dvector->GetValidity();
    // ExternalValidityMask adds no members, so this downcast only exposes
    // access to ValidityMask's protected fields.
    auto ext_validity = static_cast<ExternalValidityMask *>(&validity);

    // Use the shared_ptr aliasing constructor: the control block ref-counts the
    // ExternalVectorBuffer (preventing the Rust buffer from being freed),
    // while the stored pointer points to the explicit data_ptr.
    auto ext_buf = reinterpret_cast<shared_ptr<ExternalVectorBuffer> *>(buffer);
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

extern "C" duckdb_vx_vector_buffer duckdb_vx_vector_buffer_create(duckdb_vx_data buffer) {
    auto data = reinterpret_cast<CData *>(buffer);
    auto *shared_buffer =
        new shared_ptr<ExternalVectorBuffer>(make_shared_ptr<ExternalVectorBuffer>(unique_ptr<CData>(data)));
    return reinterpret_cast<duckdb_vx_vector_buffer>(shared_buffer);
}

extern "C" void duckdb_vx_vector_buffer_destroy(duckdb_vx_vector_buffer *buffer) {
    if (buffer != nullptr && *buffer != nullptr) {
        auto shared_buffer = reinterpret_cast<shared_ptr<ExternalVectorBuffer> *>(*buffer);
        delete shared_buffer;
        *buffer = nullptr;
    }
}
