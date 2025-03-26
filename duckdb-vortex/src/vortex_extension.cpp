#define DUCKDB_EXTENSION_MAIN

#define ENABLE_DUCKDB_FFI

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "vortex_extension.hpp"

#include "vortex.h"

#include "expr/expr.hpp"
#include "rust_vector_buffer.hpp"

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif

namespace duckdb {

/// Bind data for the Vortex table function that holds information about the
/// file and its schema. This data is populated during the bind phase, which
/// happens during the query planning phase.
struct VortexBindData : public TableFunctionData {
	string file_name;
	vector<LogicalType> columns_types;
	vector<string> column_names;
	uint64_t num_columns;
	File *file;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<VortexBindData>();
		return file_name == other.file_name && column_names == other.column_names;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<VortexBindData>();
		result->file_name = file_name;
		result->columns_types = columns_types;
		result->column_names = column_names;
		result->num_columns = num_columns;
		result->file = file;
		return std::move(result);
	}
};

/// Local state for the Vortex table function that tracks the progress of a scan
/// operation. In DuckDB's execution model, a query reading from a file can be
/// parallelized by dividing it into ranges, each handled by a different scan.
struct VortexScanLocalState : public LocalTableFunctionState {
	idx_t current_row;
	bool finished;
	Array *array;
};

struct VortexScanGlobalState : public GlobalTableFunctionState {
	ArrayStream *array_stream;
	std::mutex stream_lock;
	bool finished;

	optional_ptr<TableFilterSet> filter;
	// The column idx that must be returned by the scan.
	vector<idx_t> column_ids;

	vector<idx_t> projection_ids;

	// This is the max number threads that the extension might use.
	idx_t MaxThreads() const override {
		return 99999;
	}
};

string create_filter_expression(const VortexBindData &bind_data, VortexScanGlobalState &global_state) {
	if (global_state.filter == nullptr) {
		return "";
	}

	google::protobuf::Arena arena;
	vector<vortex::expr::Expr *> exprs;

	for (const auto &[col_id, value] : global_state.filter->filters) {
		auto col_name = bind_data.column_names[global_state.column_ids[col_id]];
		auto conj = table_expression_into_expr(arena, *value, col_name);
		exprs.push_back(conj);
	}

	auto expr = flatten_exprs(arena, exprs);
	return expr->SerializeAsString();
}

static void VortexScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<VortexBindData>();              // NOLINT
	auto &local_state = data.local_state->Cast<VortexScanLocalState>();    // NOLINT
	auto &global_state = data.global_state->Cast<VortexScanGlobalState>(); // NOLINT

	if (local_state.finished) {
		return;
	}

	if (local_state.array == nullptr) {
		std::lock_guard l(global_state.stream_lock);

		if (global_state.finished) {
			local_state.finished = true;
			return;
		}

		if (global_state.array_stream == nullptr) {
			auto column_names = std::vector<char const *>();
			for (auto col_id : global_state.projection_ids) {
				assert(col_id < bind_data.column_names.size());
				column_names.push_back(bind_data.column_names[col_id].c_str());
			}

			auto str = create_filter_expression(bind_data, global_state);

			auto options = FileScanOptions {
			    .projection = column_names.data(),
			    .projection_len = static_cast<int>(global_state.projection_ids.size()),
			    .filter_expression = str.data(),
			    .filter_expression_len = static_cast<int>(str.length()),
			    // This is a multiple of the 2048 duckdb vector size, it needs tuning
			    // This has a few factor effecting it:
			    //  1. A smaller value means for work for the vortex file reader.
			    //  2. A larger value reduces the parallelism available to the scanner
			    .split_by_row_count = 2048 * 32 * 4,
			};

			global_state.array_stream = File_scan(bind_data.file, &options);
		}

		auto next = FFIArrayStream_next(global_state.array_stream);
		if (!next) {
			FFIArrayStream_free(global_state.array_stream);
			global_state.finished = true;
			local_state.finished = true;
			return;
		}
		local_state.array = FFIArrayStream_current(global_state.array_stream);
		local_state.current_row = 0;
	}

	local_state.current_row = FFIArray_to_duckdb_chunk(local_state.array, local_state.current_row,
	                                                   reinterpret_cast<duckdb_data_chunk>(&output));

	if (local_state.current_row == 0) {
		FFIArray_free(local_state.array);
		local_state.array = nullptr;
	}
}

/// Extracts schema information from a Vortex file's data type.
static void ExtractVortexSchema(const DType *file_dtype, vector<LogicalType> &column_types,
                                vector<string> &column_names) {
	uint32_t field_count = DType_field_count(file_dtype);
	for (uint32_t idx = 0; idx < field_count; idx++) {
		char name_buffer[512];
		int name_len = 0;

		DType_field_name(file_dtype, idx, name_buffer, &name_len);
		std::string field_name(name_buffer, name_len);

		DType *field_dtype = DType_field_dtype(file_dtype, idx);
		auto duckdb_type = reinterpret_cast<LogicalType *>(DType_to_duckdb_logical_type(field_dtype));

		column_names.push_back(field_name);
		column_types.push_back(*duckdb_type);
		DType_free(field_dtype);
	}
}

std::string EnsureFileProtocol(const std::string &path) {
	const std::string prefix = "file://";

	// Check if the string already starts with "file://"
	if (path.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), path.begin())) {
		return path;
	}
	return prefix + path;
}

/// The bind function (for the Vortex table function) is called during query
/// planning. The bind phase happens once per query and allows DuckDB to know
/// the schema of the data before execution begins. This enables optimizations
/// like projection pushdown and predicate pushdown.
static unique_ptr<FunctionData> VortexBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &column_types, vector<string> &column_names) {
	auto result = make_uniq<VortexBindData>();

	// Get the filename from the input.
	auto filename = EnsureFileProtocol(input.inputs[0].GetValue<string>());
	result->file_name = filename;

	// Set up options for opening the file
	FileOpenOptions options;
	options.uri = filename.c_str();
	options.property_keys = nullptr;
	options.property_vals = nullptr;
	options.property_len = 0;

	File *file = File_open(&options);
	if (!file) {
		throw IOException("Failed to open Vortex file: " + filename);
	}

	const DType *file_dtype = File_dtype(file);
	if (DType_get(file_dtype) != DTYPE_STRUCT) {
		File_free(file);
		throw FatalException("Vortex file does not contain a struct array as a top-level dtype");
	}
	ExtractVortexSchema(file_dtype, column_types, column_names);

	result->column_names = column_names;
	result->columns_types = column_types;
	result->file = file;

	return std::move(result);
}

unique_ptr<NodeStatistics> VortexCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto data = bind_data->Cast<VortexBindData>();

	return make_uniq<NodeStatistics>(data.num_columns, data.num_columns);
}

/// Called when the extension is loaded by DuckDB.
/// It is responsible for registering functions and initializing state.
///
/// Specifically, the `read_vortex` table function enables reading data from
/// Vortex files in SQL queries.
void VortexExtension::Load(DuckDB &db) {
	DatabaseInstance &instance = *db.instance;
	TableFunction vortex_func("read_vortex", {LogicalType::VARCHAR}, VortexScanFunction, VortexBind);

	vortex_func.init_local = [](ExecutionContext &context, TableFunctionInitInput &input,
	                            GlobalTableFunctionState *global_state) -> unique_ptr<LocalTableFunctionState> {
		auto state = make_uniq<VortexScanLocalState>();

		return state;
	};

	vortex_func.init_global = [](ClientContext &context,
	                             TableFunctionInitInput &input) -> unique_ptr<GlobalTableFunctionState> {
		auto state = make_uniq<VortexScanGlobalState>();

		state->filter = input.filters;

		state->projection_ids = vector<column_t>();
		state->projection_ids.reserve(input.projection_ids.size());
		for (auto proj_id : input.projection_ids) {
			state->projection_ids.push_back(input.column_ids[proj_id]);
		}

		state->column_ids = input.column_ids;

		return state;
	};

	vortex_func.projection_pushdown = true;
	vortex_func.cardinality = VortexCardinality;
	vortex_func.filter_pushdown = true;
	vortex_func.filter_prune = true;

	ExtensionUtil::RegisterFunction(instance, vortex_func);
}

/// Returns the name of the Vortex extension.
///
/// It is used by DuckDB to identify the extension.
///
/// Example:
/// ```
/// LOAD vortex;
/// ```
std::string VortexExtension::Name() {
	return "vortex";
}

//! Returns the version of the Vortex extension.
std::string VortexExtension::Version() const {
	return "0.1.0";
}

} // namespace duckdb

extern "C" {
DUCKDB_EXTENSION_API void vortex_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::VortexExtension>();
}

DUCKDB_EXTENSION_API const char *vortex_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}
