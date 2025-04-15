#pragma once

#include "duckdb.hpp"
#include "vortex.hpp"
#include "vortex_error.hpp"

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
		FFIError *error;
		auto vx_file = duckdb::make_uniq<VortexFile>(File_open(options, &error));
		HandleError(error);
		return vx_file;
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
		FFIError *error;
		auto idx = FFIArray_to_duckdb_chunk(array, current_row, output, cache->cache, &error);
		HandleError(error);
		return idx;
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
		auto stream = FFIArrayStream_current(array_stream);
		if (stream) {
			return duckdb::make_uniq<VortexArray>(stream);
		} else {
			throw duckdb::InternalException("No more arrays in stream");
		}
	}

	bool NextArray() const {
		FFIError *error;
		auto stream = FFIArrayStream_next(array_stream, &error);
		HandleError(error);
		return stream;
	}

	ArrayStream *array_stream;
};
