#define DUCKDB_EXTENSION_MAIN

#define ENABLE_DUCKDB_FFI

#include "duckdb/main/extension_util.hpp"
#include "vortex_extension.hpp"

#include "vortex_write.hpp"
#include "vortex_scan.hpp"

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif

namespace duckdb {

/// Called when the extension is loaded by DuckDB.
/// It is responsible for registering functions and initializing state.
///
/// Specifically, the `read_vortex` table function enables reading data from
/// Vortex files in SQL queries.
void VortexExtension::Load(DuckDB &db) {
	DatabaseInstance &instance = *db.instance;

	RegisterVortexWriteFunction(instance);
	RegisterVortexScanFunction(instance);
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
