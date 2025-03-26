#include "rust_vector_buffer.hpp"
#include "duckdb/common/types/vector_buffer.hpp"
#include "duckdb/common/types/vector.hpp"

extern "C" {
CppVectorBuffer *NewCppVectorBuffer(FFIDuckDBBuffer *buffer) {
	auto rbuffer = duckdb::make_shared_ptr<RustVectorBuffer>(buffer);
	auto ribuffer = new CppVectorBufferInternal {.buffer = rbuffer};
	return reinterpret_cast<CppVectorBuffer *>(ribuffer);
}

void AssignBufferToVec(duckdb_vector vec, CppVectorBuffer *buffer) {
	auto buf = reinterpret_cast<CppVectorBufferInternal *>(buffer);
	auto dvec = reinterpret_cast<duckdb::Vector *>(vec);
	duckdb::StringVector::AddBuffer(*dvec, buf->buffer);
}
}