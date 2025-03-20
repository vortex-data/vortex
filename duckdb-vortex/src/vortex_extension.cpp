#define DUCKDB_EXTENSION_MAIN

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "vortex_extension.hpp"

#include "vortex.h"

extern "C" {
const char *vortex_duckdb_hello();
}

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

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<VortexBindData>();
		return file_name == other.file_name && column_names == other.column_names;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<VortexBindData>();
		result->file_name = file_name;
		result->columns_types = columns_types;
		result->column_names = column_names;
		return std::move(result);
	}
};

/// Local state for the Vortex table function that tracks the progress of a scan
/// operation. In DuckDB's execution model, a query reading from a file can be
/// parallelized by dividing it into ranges, each handled by a different scan.
struct VortexScanState : public LocalTableFunctionState {
	idx_t current_row = 0;
	bool finished = false;
};

static void VortexScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<VortexBindData>(); // NOLINT
	auto &state = data.local_state->Cast<VortexScanState>();

	if (state.finished) {
		return;
	}

	// TODO: Read data from file into output chunk

	// Set dummy value.
	output.SetCardinality(0);

	// When done reading, set finished to true
	state.finished = true;
}

/// Converts a Vortex data type to a DuckDB logical type.
static LogicalType VortexTypeToDuckDBType(uint8_t dtype_tag) {
	static const std::unordered_map<uint8_t, LogicalType> type_map = {
	    {DTYPE_BOOL, LogicalType::BOOLEAN},
	    {DTYPE_PRIMITIVE_I8, LogicalType::TINYINT},
	    {DTYPE_PRIMITIVE_I16, LogicalType::SMALLINT},
	    {DTYPE_PRIMITIVE_I32, LogicalType::INTEGER},
	    {DTYPE_PRIMITIVE_I64, LogicalType::BIGINT},
	    {DTYPE_PRIMITIVE_U8, LogicalType::UTINYINT},
	    {DTYPE_PRIMITIVE_U16, LogicalType::USMALLINT},
	    {DTYPE_PRIMITIVE_U32, LogicalType::UINTEGER},
	    {DTYPE_PRIMITIVE_U64, LogicalType::UBIGINT},
	    {DTYPE_PRIMITIVE_F16, LogicalType::FLOAT},
	    {DTYPE_PRIMITIVE_F32, LogicalType::FLOAT},
	    {DTYPE_PRIMITIVE_F64, LogicalType::DOUBLE},
	    {DTYPE_UTF8, LogicalType::VARCHAR},
	    {DTYPE_BINARY, LogicalType::BLOB},
	};

	auto it = type_map.find(dtype_tag);
	if (it != type_map.end()) {
		return it->second;
	}

	// For unsupported types, default to VARCHAR.
	return LogicalType::VARCHAR;
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
		LogicalType duckdb_type = VortexTypeToDuckDBType(DType_get(field_dtype));

		column_names.push_back(field_name);
		column_types.push_back(duckdb_type);

		DType_free(field_dtype);
	}
}

/// The bind function (for the Vortex table function) is called during query
/// planning. The bind phase happens once per query and allows DuckDB to know
/// the schema of the data before execution begins. This enables optimizations
/// like projection pushdown and predicate pushdown.
static unique_ptr<FunctionData> VortexBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &column_types, vector<string> &column_names) {
	auto result = make_uniq<VortexBindData>();

	// Get the filename from the input.
	auto filename = input.inputs[0].GetValue<string>();
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
	File_free(file);

	result->column_names = column_names;
	result->columns_types = column_types;

	return std::move(result);
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
		return duckdb::make_uniq<VortexScanState>();
	};

	// vortex_func.projection_pushdown = true;
	// vortex_func.filter_pushdown = true;

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
