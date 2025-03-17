#define DUCKDB_EXTENSION_MAIN

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/common/multi_file_reader_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

#include "vortex_duckdb_extension.hpp"

extern "C" {
const char *vortex_duckdb_hello();
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif

class VortexFileReaderOptions : public duckdb::BaseFileReaderOptions {
public:
	bool file_reader_flag {false};
};

struct VortexReadBindData : public duckdb::TableFunctionData {
	bool read_bind_data_flag {false};
};

struct VortexInitData : public duckdb::GlobalTableFunctionState {
	std::atomic<bool> done {false};

	static duckdb::unique_ptr<GlobalTableFunctionState> Create() {
		return duckdb::make_uniq<VortexInitData>();
	}
};

// Vortex file info implementation to integrate with DuckDBs MultiFileReader.
struct VortexMultiFileInfo {
	static void GetBindInfo(const duckdb::TableFunctionData &bind_data_p, duckdb::BindInfo &info) {
		throw std::runtime_error("called vortex get bind info func");
		// auto &bind_data = bind_data_p.Cast<VortexReadBindData>();
		// info.type = duckdb::ScanType::EXTERNAL;
		// info.InsertOption("setting", duckdb::Value::BOOLEAN(bind_data.read_bind_data_flag));
	}

	static std::unique_ptr<duckdb::BaseFileReaderOptions>
	InitializeOptions(duckdb::ClientContext &context, duckdb::optional_ptr<duckdb::TableFunctionInfo> info) {
		return duckdb::make_uniq<VortexFileReaderOptions>();
	}

	static duckdb::optional_idx MaxThreads(const duckdb::MultiFileBindData &bind_data,
	                                       const duckdb::MultiFileGlobalState &global_state,
	                                       duckdb::FileExpandResult expand_result) {
		return 1;
	}

	static std::unique_ptr<duckdb::TableFunctionData>
	InitializeBindData(duckdb::MultiFileBindData &multi_file_data,
	                   std::unique_ptr<duckdb::BaseFileReaderOptions> options_p) {
		return duckdb::make_uniq<VortexReadBindData>();
	}
};

extern "C" {
// Entrypoint to initialize the DuckDB extension calling `VortexDuckdbExtension::Load`.
DUCKDB_EXTENSION_API void vortex_duckdb_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::VortexDuckdbExtension>();
}

DUCKDB_EXTENSION_API const char *vortex_duckdb_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

void duckdb::VortexDuckdbExtension::Load(DuckDB &db) {
	DatabaseInstance &instance = *db.instance;

	// Example function to showcase an extension function.
	auto vortex_hello =
	    ScalarFunction("vortex_hello", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                   [](duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
		                   auto &name_vector = args.data[0];

		                   duckdb::UnaryExecutor::Execute<duckdb::string_t, duckdb::string_t>(
		                       name_vector, result, args.size(), [&](duckdb::string_t name) {
			                       auto str = std::string(vortex_duckdb_hello()) + name.GetString();
			                       return duckdb::StringVector::AddString(result, str);
		                       });
	                   });
	ExtensionUtil::RegisterFunction(instance, vortex_hello);

	auto table_func = [](duckdb::ClientContext &context, duckdb::TableFunctionInput &data,
	                                                duckdb::DataChunk &output) {
		throw std::runtime_error("table_func");

		auto &state = data.global_state->Cast<VortexInitData>();
		if (state.done.exchange(true)) {
			output.SetCardinality(0);
			return;
		}
	};

	auto bind_func = [](duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
	                           duckdb::vector<duckdb::LogicalType> &return_types,
	                           duckdb::vector<duckdb::string> &names) -> duckdb::unique_ptr<duckdb::FunctionData> {
		return duckdb::make_uniq<VortexReadBindData>();
	};

	auto init_func =
	    [](duckdb::ClientContext &context,
	       duckdb::TableFunctionInitInput &input) -> duckdb::unique_ptr<duckdb::GlobalTableFunctionState> {
		return duckdb::make_uniq<VortexInitData>();
	};

	duckdb::CreateTableFunctionInfo table_func_info(
	    {"vortex_table_func", {duckdb::LogicalType::VARCHAR}, table_func, bind_func, init_func});

	duckdb::ExtensionUtil::RegisterFunction(instance, table_func_info);
}

std::string duckdb::VortexDuckdbExtension::Name() {
	return "vortex_duckdb";
}

std::string duckdb::VortexDuckdbExtension::Version() const {
	return EXT_VERSION_VORTEX_DUCKDB;
}
