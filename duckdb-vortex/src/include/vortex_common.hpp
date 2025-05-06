#pragma once

#include "duckdb.hpp"
#include "vortex.hpp"
#include "vortex_error.hpp"

#include <duckdb/common/unique_ptr.hpp>

struct VortexConversionCache {
	explicit VortexConversionCache(const unsigned long cache_id) : cache(vx_conversion_cache_create(cache_id)) {
	}

	~VortexConversionCache() {
		vx_conversion_cache_free(cache);
	}

	vx_conversion_cache *cache;
};

struct VortexFileReader {
	explicit VortexFileReader(vx_file_reader *file) : file(file) {
	}

	~VortexFileReader() {
		vx_file_reader_free(file);
	}

	static duckdb::unique_ptr<VortexFileReader> Open(const vx_file_open_options *options) {
		vx_error *error;
		auto file = vx_file_open_reader(options, &error);
		HandleError(error);
		return duckdb::make_uniq<VortexFileReader>(file);
	}

	vx_file_reader *file;
};

struct VortexArray {
	explicit VortexArray(vx_array *array) : array(array) {
	}

	~VortexArray() {
		if (array != nullptr) {
			vx_array_free(array);
		}
	}

	idx_t ToDuckDBVector(idx_t current_row, duckdb_data_chunk output, const VortexConversionCache *cache) const {
		vx_error *error;
		auto idx = vx_array_to_duckdb_chunk(array, current_row, output, cache->cache, &error);
		HandleError(error);
		return idx;
	}

	vx_array *array;
};

struct VortexArrayStream {
	explicit VortexArrayStream(vx_array_stream *array_stream) : array_stream(array_stream) {
	}

	~VortexArrayStream() {
		vx_array_stream_free(array_stream);
	}

	duckdb::unique_ptr<VortexArray> NextArray() const {
		vx_error *error;
		auto array = vx_array_stream_next(array_stream, &error);
		HandleError(error);
		if (array == nullptr) {
			return nullptr;
		}
		return duckdb::make_uniq<VortexArray>(array);
	}

	vx_array_stream *array_stream;
};

struct ArrayStreamFileSink {
	explicit ArrayStreamFileSink(vx_array_stream_file_sink *sink) : sink(sink) {
	}

	ArrayStreamFileSink(std::string file_path, vx_dtype *dtype) {
	    vx_error *error = nullptr;
        sink = vx_array_stream_file_sink_open(file_path.c_str(), dtype, &error);
        HandleError(error);
	}

	void PushChunk(duckdb::DataChunk &chunk) {
		vx_error *error;
		vx_array_stream_push_duckdb_chunk(sink, reinterpret_cast<duckdb_data_chunk>(&chunk), &error);
		HandleError(error);
	}

	void Close() {
		vx_error *error;
		vx_array_stream_file_sink_close(sink, &error);
		HandleError(error);
		this->sink = nullptr;
	}

	~ArrayStreamFileSink() {
	// "should dctor a sink, before closing it
		D_ASSERT(sink == nullptr);
	}


	vx_array_stream_file_sink* sink;
};
