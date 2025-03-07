#define DUCKDB_EXTENSION_MAIN

#include "vortex_duckdb_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

inline void VortexDuckdbScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "VortexDuckdb "+name.GetString()+" 🐥");
        });
}

// inline void VortexDummyTable()

static int64_t times = 0;

static void VortexScanImplementation(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto v = Vector{LogicalType::INTEGER, 2};
	v.Reference(Value(2));

	if (times >= 3) {
		times = 0;
		return;
	}

	times++;

	output.data[0].Reference(v);
	output.SetCardinality(2);
}

struct VortexReadBindData : public TableFunctionData {};

auto result = make_uniq<VortexReadBindData>();


static unique_ptr<FunctionData> VortexScanBindImplementation(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types,	vector<string> &names) {
	names.emplace_back("int");
	return_types.emplace_back(LogicalType::INTEGER);

	return make_uniq<VortexReadBindData>();
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register a dummy table function
    auto vortex_duckdb_scalar_function = ScalarFunction("vortex_hello", {LogicalType::VARCHAR}, LogicalType::VARCHAR, VortexDuckdbScalarFun);
    ExtensionUtil::RegisterFunction(instance, vortex_duckdb_scalar_function);

	TableFunction table_function("vortex_scan", {LogicalType::VARCHAR}, VortexScanImplementation, VortexScanBindImplementation);
	// table_function.statistics = ParquetScanStats;
	// table_function.cardinality = ParquetCardinality;
	// table_function.table_scan_progress = ParquetProgress;
	// table_function.named_parameters["binary_as_string"] = LogicalType::BOOLEAN;
	// table_function.named_parameters["file_row_number"] = LogicalType::BOOLEAN;
	// table_function.named_parameters["debug_use_openssl"] = LogicalType::BOOLEAN;
	// table_function.named_parameters["compression"] = LogicalType::VARCHAR;
	// table_function.named_parameters["explicit_cardinality"] = LogicalType::UBIGINT;
	// table_function.named_parameters["schema"] = LogicalTypeId::ANY;
	// table_function.named_parameters["encryption_config"] = LogicalTypeId::ANY;
	// table_function.named_parameters["parquet_version"] = LogicalType::VARCHAR;
	// table_function.get_partition_data = ParquetScanGetPartitionData;
	// table_function.serialize = ParquetScanSerialize;
	// table_function.deserialize = ParquetScanDeserialize;
	// table_function.get_bind_info = ParquetGetBindInfo;
	// table_function.projection_pushdown = true;
	// table_function.filter_pushdown = true;
	// table_function.filter_prune = true;
	// table_function.pushdown_complex_filter = ParquetComplexFilterPushdown;
	// table_function.get_partition_info = ParquetGetPartitionInfo;
	// table_function.arguments[0] = LogicalType::LIST(LogicalType::VARCHAR);


	// named_parameter_map_t named_parameters({{"binary_as_string", Value::BOOLEAN(binary_as_string)}});
	// return TableFunction("parquet_scan", params, named_parameters)->Alias(parquet_file);

	ExtensionUtil::RegisterFunction(instance, table_function);

}

void VortexDuckdbExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string VortexDuckdbExtension::Name() {
	return "vortex_duckdb";
}

std::string VortexDuckdbExtension::Version() const {
#ifdef EXT_VERSION_VORTEX_DUCKDB
	return EXT_VERSION_VORTEX_DUCKDB;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void vortex_duckdb_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::VortexDuckdbExtension>();
}

DUCKDB_EXTENSION_API const char *vortex_duckdb_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
