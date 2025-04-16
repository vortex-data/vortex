#define ENABLE_DUCKDB_FFI

#include "vortex_write.hpp"
#include "vortex_common.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/function/copy_function.hpp"

namespace duckdb {

struct VortexWriteBindData : public TableFunctionData {
	vector<LogicalType> sql_types;
	vector<string> column_names;
};

struct VortexWriteGlobalData : public GlobalFunctionData {
	std::string file_name;
	std::unique_ptr<VortexFileReader> file;
	unique_ptr<VortexArray> array;
};

struct VortexWriteLocalData : public LocalFunctionData {};

void VortexWriteSink(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                     LocalFunctionData &lstate, DataChunk &input) {
	auto &global_state = gstate.Cast<VortexWriteGlobalData>();
	auto bind = bind_data.Cast<VortexWriteBindData>();

	auto chunk = DataChunk();
	chunk.Initialize(Allocator::Get(context.client), bind.sql_types);

	for (auto i = 0u; i < input.ColumnCount(); i++) {
		input.data[i].Flatten(input.size());
	}

	auto new_array =
	    vx_array_append_duckdb_chunk(global_state.array->array, reinterpret_cast<duckdb_data_chunk>(&input));
	global_state.array = make_uniq<VortexArray>(new_array);
}

void RegisterVortexWriteFunction(DatabaseInstance &instance) {
	CopyFunction function("vortex");
	function.copy_to_bind = [](ClientContext &context, CopyFunctionBindInput &input, const vector<string> &names,
	                           const vector<LogicalType> &sql_types) -> unique_ptr<FunctionData> {
		auto result = make_uniq<VortexWriteBindData>();
		result->sql_types = sql_types;
		result->column_names = names;
		return std::move(result);
	};
	function.copy_to_initialize_global = [](ClientContext &context, FunctionData &bind_data,
	                                        const string &file_path) -> unique_ptr<GlobalFunctionData> {
		auto &bind = bind_data.Cast<VortexWriteBindData>();
		auto gstate = make_uniq<VortexWriteGlobalData>();
		gstate->file_name = file_path;

		auto column_names = std::vector<const char *>();
		for (const auto &col_id : bind.column_names) {
			column_names.push_back(col_id.c_str());
		}

		auto column_types = std::vector<duckdb_logical_type>();
		for (auto &col_type : bind.sql_types) {
			column_types.push_back(reinterpret_cast<duckdb_logical_type>(&col_type));
		}
		vx_error *error = nullptr;
		auto array = vx_array_create_empty_from_duckdb_table(column_types.data(), column_names.data(),
		                                                     column_names.size(), &error);
		HandleError(error);

		gstate->array = make_uniq<VortexArray>(array);
		return std::move(gstate);
	};
	function.copy_to_initialize_local = [](ExecutionContext &context,
	                                       FunctionData &bind_data) -> unique_ptr<LocalFunctionData> {
		return std::move(make_uniq<VortexWriteLocalData>());
	};
	function.copy_to_sink = VortexWriteSink;
	function.copy_to_finalize = [](ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate) {
		auto &global_state = gstate.Cast<VortexWriteGlobalData>();
		auto opts = vx_file_create_options();
		opts.path = global_state.file_name.c_str();
		vx_error *error;
		auto file = vx_file_create(&opts, &error);
		if (file == nullptr) {
			HandleError(error);
		}
		vx_file_write_array(file, global_state.array->array, &error);
		HandleError(error);
		vx_file_writer_free(file);
	};
	function.execution_mode = [](bool preserve_insertion_order,
	                             bool supports_batch_index) -> CopyFunctionExecutionMode {
		return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
	};
	function.extension = "vortex";

	ExtensionUtil::RegisterFunction(instance, function);
}

} // namespace duckdb