#pragma once

#include "duckdb.hpp"

#ifndef ENABLE_DUCKDB_FFI
#define ENABLE_DUCKDB_FFI
#endif

#include "vortex.h"

struct VortexConversionCache {
	explicit VortexConversionCache(const unsigned long cache_id) : cache(ConversionCache_create(cache_id)) {
	}

	~VortexConversionCache() {
		ConversionCache_free(cache);
	}

	FFIConversionCache *cache;
};

struct VortexFile {
	explicit VortexFile(File *file) : file(file) {
	}

	~VortexFile() {
		File_free(file);
	}

	static duckdb::unique_ptr<VortexFile> Open(const FileOpenOptions *options) {
		return duckdb::make_uniq<VortexFile>(File_open(options));
	}

	File *file;
};

struct VortexArray {
	explicit VortexArray(Array *array) : array(array) {
	}

	~VortexArray() {
		FFIArray_free(array);
	}

	idx_t ToDuckDBVector(idx_t current_row, duckdb_data_chunk output, const VortexConversionCache *cache) const {
		return FFIArray_to_duckdb_chunk(array, current_row, output, cache->cache);
	}

	Array *array;
};

struct VortexArrayStream {
	explicit VortexArrayStream(ArrayStream *array_stream) : array_stream(array_stream) {
	}

	~VortexArrayStream() {
		FFIArrayStream_free(array_stream);
	}

	duckdb::unique_ptr<VortexArray> CurrentArray() const {
		return duckdb::make_uniq<VortexArray>(FFIArrayStream_current(array_stream));
	}

	bool NextArray() const {
		return FFIArrayStream_next(array_stream);
	}

	ArrayStream *array_stream;
};
