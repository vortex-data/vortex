#define ENABLE_DUCKDB_FFI

#include "vortex_write.hpp"
#include "vortex_common.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"

// TODO(joe): enable multi-threaded writes, see `VortexWriteSink`.

namespace duckdb {

struct VortexWriteBindData : public TableFunctionData {
	//! True is the column is nullable
	vector<unsigned char> column_nullable;

	vector<LogicalType> sql_types;
	vector<string> column_names;
};

struct VortexWriteGlobalData : public GlobalFunctionData {
	unique_ptr<ArrayStreamSink> sink;
};

struct VortexWriteLocalData : public LocalFunctionData {};

void VortexWriteSink(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                     LocalFunctionData &lstate, DataChunk &input) {
	auto &global_state = gstate.Cast<VortexWriteGlobalData>();
	auto bind = bind_data.Cast<VortexWriteBindData>();

	for (auto i = 0u; i < input.ColumnCount(); i++) {
		input.data[i].Flatten(input.size());
	}
	// TODO(joe): go to a model of combining local chunked into arrays of a specific size
	// before push each of these larger chunks into the global_state
	global_state.sink->PushChunk(input);
}

std::vector<idx_t> TableNullability(ClientContext &context, const string &catalog_name, const string &schema,
                                    const string &table) {
	auto &catalog = Catalog::GetCatalog(context, catalog_name);

	QueryErrorContext error_context;
	// Main is the default schema
	auto schema_name = schema != "" ? schema : "main";

	auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema_name, table, OnEntryNotFound::RETURN_NULL,
	                              error_context);
	auto vec = std::vector<idx_t>();
	if (!entry) {
		// If there is no entry, it is okay to return all nullable columns.
		return vec;
	}

	auto &table_entry = entry->Cast<TableCatalogEntry>();
	for (auto &constraint : table_entry.GetConstraints()) {
		if (constraint->type == ConstraintType::NOT_NULL) {
			auto &null_constraint = constraint->Cast<NotNullConstraint>();
			vec.push_back(null_constraint.index.index);
		}
	}
	return vec;
}

void RegisterVortexWriteFunction(DatabaseInstance &instance) {
	CopyFunction function("vortex");
	function.copy_to_bind = [](ClientContext &context, CopyFunctionBindInput &input, const vector<string> &names,
	                           const vector<LogicalType> &sql_types) -> unique_ptr<FunctionData> {
		auto result = make_uniq<VortexWriteBindData>();

		auto not_null = TableNullability(context, input.info.catalog, input.info.schema, input.info.table);

		result->column_nullable = std::vector<unsigned char>(names.size(), true);
		for (auto not_null_idx : not_null) {
			result->column_nullable[not_null_idx] = false;
		}

		result->sql_types = sql_types;
		result->column_names = names;
		return std::move(result);
	};
	function.copy_to_initialize_global = [](ClientContext &context, FunctionData &bind_data,
	                                        const string &file_path) -> unique_ptr<GlobalFunctionData> {
		auto &bind = bind_data.Cast<VortexWriteBindData>();
		auto gstate = make_uniq<VortexWriteGlobalData>();

		auto column_names = std::vector<const char *>();
		for (const auto &col_id : bind.column_names) {
			column_names.push_back(col_id.c_str());
		}

		auto column_types = std::vector<duckdb_logical_type>();
		for (auto &col_type : bind.sql_types) {
			column_types.push_back(reinterpret_cast<duckdb_logical_type>(&col_type));
		}

		auto dtype = DType::FromDuckDBTable(column_types, bind.column_nullable, column_names);
		gstate->sink = ArrayStreamSink::Create(file_path, std::move(dtype));
		return std::move(gstate);
	};
	function.copy_to_initialize_local = [](ExecutionContext &context,
	                                       FunctionData &bind_data) -> unique_ptr<LocalFunctionData> {
		return std::move(make_uniq<VortexWriteLocalData>());
	};
	function.copy_to_sink = VortexWriteSink;
	function.copy_to_finalize = [](ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate) {
		auto &global_state = gstate.Cast<VortexWriteGlobalData>();
		global_state.sink->Close();
	};
	function.execution_mode = [](bool preserve_insertion_order,
	                             bool supports_batch_index) -> CopyFunctionExecutionMode {
		return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
	};
	function.extension = "vortex";

	ExtensionUtil::RegisterFunction(instance, function);
}

} // namespace duckdb
