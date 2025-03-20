#define DUCKDB_EXTENSION_MAIN

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "vortex_extension.hpp"

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

    // TODO
    // - Open the file
    // - Read its schema
    // - Define return_types and names based on the schema

    // Set a dummy schema.
    column_names.push_back("vortex_sample_column");
    column_types.push_back(LogicalType::VARCHAR);

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
