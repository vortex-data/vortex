#pragma once
#define ENABLE_DUCKDB_FFI

#include "duckdb.hpp"
#include "duckdb/common/unique_ptr.hpp"

#include "vortex.hpp"
#include "vortex_error.hpp"
#include "vortex_session.hpp"

namespace vortex {

struct DType {
	explicit DType(const vx_dtype *dtype) : dtype(dtype) {
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

	const vx_dtype *dtype;
};

struct VortexFile {
	explicit VortexFile(const vx_file *file) : file(file) {
	}

	~VortexFile() {
		vx_file_free(file);
	}

	static duckdb::unique_ptr<VortexFile> Open(const vx_file_open_options *options, VortexSession &session) {
		auto file = Try([&](auto err) { return vx_file_open_reader(options, session.session, err); });
		return duckdb::make_uniq<VortexFile>(file);
	}

	vx_array_iterator *Scan(const vx_file_scan_options *options) {
		return Try([&](auto err) { return vx_file_scan(this->file, options, err); });
	}

	bool CanPrune(const char *filter_expression, unsigned int filter_expression_len, unsigned long file_idx) {
		return Try([&](auto err) {
			return vx_file_can_prune(this->file, filter_expression, filter_expression_len, file_idx, err);
		});
	}

	uint64_t RowCount() {
		return vx_file_row_count(file);
	}

	struct DType DType() {
		return vortex::DType(vx_dtype_clone(vx_file_dtype(file)));
	}

	const vx_file *file;
};


struct Array {
	explicit Array(const vx_array *array) : array(array) {
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

	const vx_array *array;
};


struct ArrayIterator {
	explicit ArrayIterator(vx_array_iterator *array_iter) : array_iter(array_iter) {
	}

	/// Releases ownership of the native array iterator ptr to the caller. The caller is then responsible for
	/// eventually calling vx_array_iter_free.
	///
	/// This ArrayIterator is useless after this call.
	vx_array_iterator* release() {
		auto* ptr = array_iter;
		array_iter = nullptr;  // Give up ownership
		return ptr;
	}

	~ArrayIterator() {
		if (array_iter) {
			vx_array_iterator_free(array_iter);
		}
	}

	duckdb::unique_ptr<Array> NextArray() const {
		auto array = Try([&](auto err) { return vx_array_iterator_next(array_iter, err); });

		if (array == nullptr) {
			return nullptr;
		}

		return duckdb::make_uniq<Array>(array);
	}

	vx_array_iterator *array_iter;
};


struct ArrayExporter {
	explicit ArrayExporter(vx_duckdb_exporter *exporter) : exporter(exporter) {
	}

	~ArrayExporter() {
		if (exporter != nullptr) {
			vx_duckdb_exporter_free(exporter);
		}
	}

	static duckdb::unique_ptr<ArrayExporter> FromArrayIterator(duckdb::unique_ptr<ArrayIterator> array_iter) {
		auto exporter = vx_duckdb_exporter_new(array_iter->release());
		return duckdb::make_uniq<ArrayExporter>(exporter);
	}

	bool ExportNext(duckdb_data_chunk output) const {
		return Try([&](auto err) { return vx_duckdb_exporter_next(exporter, output, err); });
	}

	vx_duckdb_exporter *exporter;
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
