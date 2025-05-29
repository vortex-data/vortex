#pragma once
#define ENABLE_DUCKDB_FFI

#include "duckdb.hpp"
#include "duckdb/common/unique_ptr.hpp"

#include "vortex.hpp"
#include "vortex_error.hpp"
#include "vortex_session.hpp"

namespace vortex {

struct DType {
	explicit DType(vx_dtype *dtype) : dtype(dtype) {
	}

	static duckdb::unique_ptr<DType> FromDuckDBTable(const std::vector<duckdb_logical_type> &column_types,
	                                                 const std::vector<unsigned char> &column_nullable,
	                                                 const std::vector<const char *> &column_names) {
		D_ASSERT(column_names.size() == column_nullable.size());
		D_ASSERT(column_names.size() == column_types.size());

		auto dtype = Try([&](auto err) {
			return vx_duckdb_logical_type_to_dtype(column_types.data(), column_nullable.data(), column_names.data(),
			                                       column_names.size(), err);
		});

		return duckdb::make_uniq<DType>(dtype);
	}

	~DType() {
		if (dtype != nullptr) {
			vx_dtype_free(dtype);
		}
	}

	vx_dtype *dtype;
};

struct ConversionCache {
	explicit ConversionCache(const unsigned long cache_id) : cache(vx_conversion_cache_create(cache_id)) {
	}

	~ConversionCache() {
		vx_conversion_cache_free(cache);
	}

	vx_conversion_cache *cache;
};

struct FileReader {
	explicit FileReader(vx_file_reader *file) : file(file) {
	}

	~FileReader() {
		vx_file_reader_free(file);
	}

	static duckdb::unique_ptr<FileReader> Open(const vx_file_open_options *options, VortexSession &session) {
		auto file = Try([&](auto err) { return vx_file_open_reader(options, session.session, err); });
		return duckdb::make_uniq<FileReader>(file);
	}

	vx_array_iterator *Scan(const vx_file_scan_options *options) {
		return Try([&](auto err) { return vx_file_reader_scan(this->file, options, err); });
	}

	bool CanPrune(const char *filter_expression, unsigned int filter_expression_len) {
		return Try([&](auto err) {
			return vx_file_reader_can_prune(this->file, filter_expression, filter_expression_len, err);
		});
	}

	uint64_t FileRowCount() {
		return Try([&](auto err) { return vx_file_row_count(file, err); });
	}

	struct DType DType() {
		return vortex::DType(vx_file_dtype(file));
	}

	vx_file_reader *file;
};

struct Array {
	explicit Array(vx_array *array) : array(array) {
	}

	~Array() {
		if (array != nullptr) {
			vx_array_free(array);
		}
	}

	static duckdb::unique_ptr<Array> FromDuckDBChunk(DType &dtype, duckdb::DataChunk &chunk) {
		auto array = Try([&](auto err) {
			return vx_duckdb_chunk_to_array(reinterpret_cast<duckdb_data_chunk>(&chunk), dtype.dtype, err);
		});

		return duckdb::make_uniq<Array>(array);
	}

	idx_t ToDuckDBVector(idx_t current_row, duckdb_data_chunk output, const ConversionCache *cache) const {
		return Try([&](auto err) { return vx_array_to_duckdb_chunk(array, current_row, output, cache->cache, err); });
	}

	vx_array *array;
};

struct ArrayIterator {
	explicit ArrayIterator(vx_array_iterator *array_iter) : array_iter(array_iter) {
	}

	~ArrayIterator() {
		vx_array_iter_free(array_iter);
	}

	duckdb::unique_ptr<Array> NextArray() const {
		auto array = Try([&](auto err) { return vx_array_iter_next(array_iter, err); });

		if (array == nullptr) {
			return nullptr;
		}

		return duckdb::make_uniq<Array>(array);
	}

	vx_array_iterator *array_iter;
};

struct ArrayStreamSink {
	explicit ArrayStreamSink(vx_array_sink *sink, duckdb::unique_ptr<DType> dtype)
	    : sink(sink), dtype(std::move(dtype)) {
	}

	static duckdb::unique_ptr<ArrayStreamSink> Create(std::string file_path, duckdb::unique_ptr<DType> &&dtype) {
		auto sink = Try([&](auto err) { return vx_array_sink_open_file(file_path.c_str(), dtype->dtype, err); });
		return duckdb::make_uniq<ArrayStreamSink>(sink, std::move(dtype));
	}

	void PushChunk(duckdb::DataChunk &chunk) {
		auto array = Array::FromDuckDBChunk(*dtype, chunk);
		Try([&](auto err) { vx_array_sink_push(sink, array->array, err); });
	}

	void Close() {
		Try([&](auto err) { vx_array_sink_close(sink, err); });
		this->sink = nullptr;
	}

	~ArrayStreamSink() {
		// "should dctor a sink, before closing it
		// If you throw during writes then the stack will be unwound and the destructor is going to be called before the
		// close method is invoked thus triggering following assertion failure and will clobber the exception printing
		// D_ASSERT(sink == nullptr);
	}

	vx_array_sink *sink;
	duckdb::unique_ptr<DType> dtype;
};

} // namespace vortex
