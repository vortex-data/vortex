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

// rust have buffer BufferFFI {arc: Arc<Buffer>}, decrement_arc_ptr(ptr);
//  let buffer = create_wrapper_buffer(ptr *mut BufferFFI) from c
//  vector.assign_buffer(buffer)  <------ c. duckdb_assign_string_buffer(buffer)
//