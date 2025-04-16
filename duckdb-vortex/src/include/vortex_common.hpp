#pragma once

#include "duckdb.hpp"
#include "vortex.hpp"
#include "vortex_error.hpp"

struct VortexConversionCache {
	explicit VortexConversionCache(const unsigned long cache_id) : cache(vx_conversion_cache_create(cache_id)) {
	}

	~VortexConversionCache() {
		vx_conversion_cache_free(cache);
	}

	VXConversionCache *cache;
};

struct VortexFile {
	explicit VortexFile(VXFile *file) : file(file) {
	}

	~VortexFile() {
		vx_file_free(file);
	}

	static duckdb::unique_ptr<VortexFile> Open(const VXFileOpenOptions *options) {
		VXError *error;
		auto vx_file = duckdb::make_uniq<VortexFile>(vx_file_open(options, &error));
		HandleError(error);
		return vx_file;
	}

	VXFile *file;
};

struct VortexArray {
	explicit VortexArray(VXArray *array) : array(array) {
	}

	~VortexArray() {
		vx_array_free(array);
	}

	idx_t ToDuckDBVector(idx_t current_row, duckdb_data_chunk output, const VortexConversionCache *cache) const {
		VXError *error;
		auto idx = vx_array_to_duckdb_chunk(array, current_row, output, cache->cache, &error);
		HandleError(error);
		return idx;
	}

	VXArray *array;
};

struct VortexArrayStream {
	explicit VortexArrayStream(VXArrayStream *array_stream) : array_stream(array_stream) {
	}

	~VortexArrayStream() {
		vx_array_stream_free(array_stream);
	}

	duckdb::unique_ptr<VortexArray> CurrentArray() const {
		auto stream = vx_array_stream_current(array_stream);
		if (stream) {
			return duckdb::make_uniq<VortexArray>(stream);
		} else {
			throw duckdb::InternalException("No more arrays in stream");
		}
	}

	bool NextArray() const {
		VXError *error;
		auto stream = vx_array_stream_next(array_stream, &error);
		HandleError(error);
		return stream;
	}

	VXArrayStream *array_stream;
};
