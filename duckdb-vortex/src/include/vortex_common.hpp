#pragma once

#include "duckdb.hpp"
#include "vortex.hpp"
#include "vortex_error.hpp"

#include <duckdb/common/unique_ptr.hpp>

struct DType {
	explicit DType(vx_dtype *dtype): dtype(dtype) {}

	static duckdb::unique_ptr<DType> FromDuckDBTable(
		const std::vector<duckdb_logical_type> &column_types,
		const std::vector<unsigned char> &column_nullable,
		const std::vector<const char *> &column_names
	) {
		D_ASSERT(column_names.size() == column_nullable.size());
		D_ASSERT(column_names.size() == column_types.size());

		vx_error *error = nullptr;
		auto dtype = vx_duckdb_logical_type_to_dtype(
			column_types.data(),
			column_nullable.data(),
			column_names.data(),
			column_names.size(),
			&error
		);
		HandleError(error);

		return duckdb::make_uniq<DType>(dtype);
	}


	~DType() {
		if (dtype != nullptr) {
			vx_dtype_free(dtype);
		}
	}

	vx_dtype *dtype;
};

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

	uint64_t FileRowCount() {
		return Try([&](auto err) { return vx_file_row_count(file, err); });
	}

	struct DType DType() {
		return ::DType(vx_file_dtype(file));
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

	static duckdb::unique_ptr<VortexArray> FromDuckDBChunk(DType &dtype, duckdb::DataChunk &chunk) {
		vx_error *error;
		auto array = vx_duckdb_chunk_to_array(reinterpret_cast<duckdb_data_chunk>(&chunk), dtype.dtype, &error);
		HandleError(error);
		return duckdb::make_uniq<VortexArray>(array);
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

struct ArrayStreamSink {
	explicit ArrayStreamSink(vx_array_sink *sink, duckdb::unique_ptr<DType> dtype) : sink(sink), dtype(std::move(dtype)) {
	}

	static duckdb::unique_ptr<ArrayStreamSink> Create(std::string file_path, duckdb::unique_ptr<DType> &&dtype) {
	    vx_error *error = nullptr;
        auto sink = vx_array_sink_open_file(file_path.c_str(), dtype->dtype, &error);
        HandleError(error);

        return duckdb::make_uniq<ArrayStreamSink>(sink, std::move(dtype));
	}

	void PushChunk(duckdb::DataChunk &chunk) {
		vx_error *error = nullptr;
		auto array = VortexArray::FromDuckDBChunk(*dtype, chunk);
		vx_array_sink_push(sink, array->array, &error);
		HandleError(error);
	}

	void Close() {
		vx_error *error;
		vx_array_sink_close(sink, &error);
		HandleError(error);

		this->sink = nullptr;
	}

	~ArrayStreamSink() {
		// "should dctor a sink, before closing it
		D_ASSERT(sink == nullptr);
	}


	vx_array_sink *sink;
	duckdb::unique_ptr<DType> dtype;
};
