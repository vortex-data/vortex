#define DUCKDB_EXTENSION_MAIN

#include "vortex_duckdb_extension.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"

extern "C" {
const char *vortex_duckdb_hello();
}

namespace duckdb {

inline void VortexDuckdbScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		auto c_str = vortex_duckdb_hello();
		auto str = std::string(c_str) + name.GetString() + " 🐥";
		return StringVector::AddString(result, str);
	});
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register a dummy table function
	auto vortex_duckdb_scalar_function =
	    ScalarFunction("vortex_hello", {LogicalType::VARCHAR}, LogicalType::VARCHAR, VortexDuckdbScalarFun);
	ExtensionUtil::RegisterFunction(instance, vortex_duckdb_scalar_function);
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
