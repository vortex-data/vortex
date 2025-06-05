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

// This is a wrapper around an externally managed buffer, which can be assigned to a Vector and
// freed once the vector is done with the buffer.
class ExternalVectorBuffer : public VectorBuffer {
	unique_ptr<CData> data;
};

extern "C" void duckdb_vx_string_vector_add_buffer(duckdb_vector ffi_vector, duckdb_vx_data buffer) {
	auto vector = reinterpret_cast<Vector *>(ffi_vector);
	auto data = reinterpret_cast<CData *>(buffer);
	auto external_buffer = duckdb::make_shared_ptr<ExternalVectorBuffer>(unique_ptr<CData>(data));
	StringVector::AddBuffer(*vector, external_buffer);
}
