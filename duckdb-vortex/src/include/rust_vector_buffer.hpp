#pragma once
#include <duckdb.h>
#include <duckdb/common/types/vector_buffer.hpp>

extern "C" {

typedef struct FFIDuckDBBuffer FFIDuckDBBuffer;

void FFIDuckDBBuffer_free(FFIDuckDBBuffer *);

}

class RustVectorBuffer : public duckdb::VectorBuffer {
public:
	RustVectorBuffer(FFIDuckDBBuffer *wrapper) : wrapper(wrapper) {
	}

	~RustVectorBuffer() override {
		FFIDuckDBBuffer_free(wrapper);
	}

private:
	FFIDuckDBBuffer *wrapper;
};

// These structs and functions are used the wrap a buffer in C++.
// See convert/array/varbinview::to_duckdb for more
extern "C" {

typedef struct {
	void *ptr;
} CppVectorBuffer;

typedef struct {
	duckdb::buffer_ptr<RustVectorBuffer> buffer;
} CppVectorBufferInternal;
CppVectorBuffer *NewCppVectorBuffer(FFIDuckDBBuffer *buffer);

void AssignBufferToVec(duckdb_vector vec, CppVectorBuffer *buffer);

}
