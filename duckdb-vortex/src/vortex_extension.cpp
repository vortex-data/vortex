#define DUCKDB_EXTENSION_MAIN

#define ENABLE_DUCKDB_FFI

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "vortex_extension.hpp"

#include "vortex_common.hpp"
#include "expr/expr.hpp"

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif

namespace duckdb {

// A value large enough that most systems can use all their threads.
// This is used to allocate `file_slots`, we could remove this later by having the init_local method increase the
// file slots size for each running.
constexpr uint32_t MAX_THREAD_COUNT = 192;

/// Bind data for the Vortex table function that holds information about the
/// file and its schema. This data is populated during the bind phase, which
/// happens during the query planning phase.
struct VortexBindData : public TableFunctionData {
	vector<LogicalType> columns_types;
	vector<string> column_names;
	uint64_t num_columns;
	unique_ptr<VortexFile> initial_file;

	shared_ptr<MultiFileList> file_list;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<VortexBindData>();
		return file_list == other.file_list && column_names == other.column_names &&
		       columns_types == other.columns_types && num_columns == other.num_columns;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<VortexBindData>();
		result->file_list = file_list;
		result->columns_types = columns_types;
		result->column_names = column_names;
		result->num_columns = num_columns;
		return std::move(result);
	}
};

/// Local state for the Vortex table function that tracks the progress of a scan
/// operation. In DuckDB's execution model, a query reading from a file can be
/// parallelized by dividing it into ranges, each handled by a different scan.
struct VortexScanLocalState : public LocalTableFunctionState {
	idx_t current_row;
	bool finished;
	unique_ptr<VortexArray> array;
	unique_ptr<VortexConversionCache> cache;
	uint32_t thread_id;

	explicit VortexScanLocalState(uint32_t thread_id)
	    : current_row(0), finished(false), array(nullptr), cache(nullptr), thread_id(thread_id) {
	}
};

struct FileSlot {
	std::mutex slot_lock;
	unique_ptr<VortexArrayStream> array_stream;
};

struct VortexScanGlobalState : public GlobalTableFunctionState {
	// Must be <= MAX_THREAD_COUNT.
	std::atomic_uint32_t thread_id_counter;
	std::atomic_bool finished;

	std::uint64_t cache_id;

	// Each thread owns a file slot and is the thing only one allowed to modify the slot itself.
	// Other threads can work-steal array batches from the slot, by taking out the mutex in the FileSlot.
	// We allocate MAX_THREAD_COUNT threads, the max number threads allowed by this extension.
	std::array<FileSlot, MAX_THREAD_COUNT> file_slots;

	std::atomic_uint32_t next_file;
	vector<string> expanded_files;

	optional_ptr<TableFilterSet> filter;

	// The precomputed filter string used in the query
	std::string filter_str;
	// The precomputed column names used in the query
	std::vector<char const *> projected_column_names;

	// The column idx that must be returned by the scan.
	vector<idx_t> column_ids;
	vector<idx_t> projection_ids;

	// This is the max number threads that the extension might use.
	idx_t MaxThreads() const override {
		return MAX_THREAD_COUNT;
	}

	explicit VortexScanGlobalState()
	    : thread_id_counter(0), finished(false), cache_id(0), file_slots(), next_file(0), filter(nullptr) {
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

static unique_ptr<VortexFile> OpenFile(const std::string &filename, vector<LogicalType> &column_types,
                                       vector<string> &column_names) {
	FileOpenOptions options;
	options.uri = filename.c_str();
	options.property_keys = nullptr;
	options.property_vals = nullptr;
	options.property_len = 0;

	auto file = VortexFile::Open(&options);
	if (!file) {
		throw IOException("Failed to open Vortex file: " + filename);
	}

	// This Ptr is owned by the file
	const DType *file_dtype = File_dtype(file->file);
	if (DType_get(file_dtype) != DTYPE_STRUCT) {
		File_free(file->file);
		throw FatalException("Vortex file does not contain a struct array as a top-level dtype");
	}

	ExtractVortexSchema(file_dtype, column_types, column_names);

	return file;
}

static void VerifyNewFile(const VortexBindData &bind_data, vector<LogicalType> &column_types,
                          vector<string> &column_names) {
	if (column_types.size() != bind_data.columns_types.size() || column_names != bind_data.column_names) {
		throw FatalException("Vortex file does not contain the same number of columns as the first");
	}
	for (auto idx = 0; idx < bind_data.columns_types.size(); idx++) {
		auto col_name = bind_data.column_names[idx];
		auto col_type = bind_data.columns_types[idx];
		if (col_name != column_names[idx]) {
			throw FatalException("Vortex file contains a column with a different name to the first");
		}
		if (col_type != column_types[idx]) {
			throw FatalException("Vortex file contains a column with a different type to the first");
		}
	}
}

static unique_ptr<VortexFile> OpenFileAndVerify(const std::string &filename, const VortexBindData &bind_data) {
	auto new_column_names = vector<string>();
	new_column_names.reserve(bind_data.column_names.size());
	auto new_column_types = vector<LogicalType>();
	new_column_names.reserve(bind_data.columns_types.size());

	auto file = OpenFile(EnsureFileProtocol(filename), new_column_types, new_column_names);
	VerifyNewFile(bind_data, new_column_types, new_column_names);
	return file;
}

static unique_ptr<VortexArrayStream> OpenArrayStream(const VortexBindData &bind_data,
                                                     VortexScanGlobalState &global_state, VortexFile *file) {
	auto options = FileScanOptions {
	    .projection = global_state.projected_column_names.data(),
	    .projection_len = static_cast<int>(global_state.projected_column_names.size()),
	    .filter_expression = global_state.filter_str.data(),
	    .filter_expression_len = static_cast<int>(global_state.filter_str.length()),
	    // This is a multiple of the 2048 duckdb vector size, it needs tuning
	    // This has a few factor effecting it:
	    //  1. A smaller value means for work for the vortex file reader.
	    //  2. A larger value reduces the parallelism available to the scanner
	    .split_by_row_count = 2048 * 32,
	};

	return make_uniq<VortexArrayStream>(File_scan(file->file, &options));
}

static void VortexScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<VortexBindData>();              // NOLINT
	auto &global_state = data.global_state->Cast<VortexScanGlobalState>(); // NOLINT
	auto &local_state = data.local_state->Cast<VortexScanLocalState>();    // NOLINT

	if (local_state.array == nullptr) {
		auto &slot = global_state.file_slots[local_state.thread_id];
		std::lock_guard _l(slot.slot_lock);

		if (global_state.finished.load() && slot.array_stream == nullptr) {
			output.SetCardinality(0);
			return;
		}

		// 1. check we can make progress on current owned file
		// 2. check we can get another file
		// todo: 3. check if we can work steal from another thread
		// 4. we are done

		auto next = slot.array_stream != nullptr ? slot.array_stream->NextArray() : false;
		while (!next) {
			if (slot.array_stream != nullptr) {
				slot.array_stream = nullptr;
			}

			auto file_idx = global_state.next_file.fetch_add(1);

			if (file_idx >= global_state.expanded_files.size()) {
				local_state.finished = true;
				global_state.finished = true;
				output.Reset();
				return;
			}

			auto file_name = global_state.expanded_files[file_idx];
			auto file = OpenFileAndVerify(file_name, bind_data);

			slot.array_stream = OpenArrayStream(bind_data, global_state, file.get());
			next = slot.array_stream->NextArray();
		}
		local_state.array = slot.array_stream->CurrentArray();
		local_state.current_row = 0;
	}

	if (local_state.cache == nullptr) {
		// Create a unique value so each cache can be differentiated.
		local_state.cache = make_uniq<VortexConversionCache>(global_state.cache_id++);
	}

	local_state.current_row = local_state.array->ToDuckDBVector(
	    local_state.current_row, reinterpret_cast<duckdb_data_chunk>(&output), local_state.cache.get());

	if (local_state.current_row == 0) {
		local_state.array = nullptr;
		local_state.cache = nullptr;
	}
}

/// The bind function (for the Vortex table function) is called during query
/// planning. The bind phase happens once per query and allows DuckDB to know
/// the schema of the data before execution begins. This enables optimizations
/// like projection pushdown and predicate pushdown.
static unique_ptr<FunctionData> VortexBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &column_types, vector<string> &column_names) {
	auto result = make_uniq<VortexBindData>();

	// Get the filename glob from the input.
	auto vec = duckdb::vector<string> {input.inputs[0].GetValue<string>()};
	result->file_list = make_shared_ptr<GlobMultiFileList>(context, vec, FileGlobOptions::DISALLOW_EMPTY);

	auto filename = EnsureFileProtocol(result->file_list->GetFirstFile());

	result->initial_file = OpenFile(filename, column_types, column_names);

	result->column_names = column_names;
	result->columns_types = column_types;

	return std::move(result);
}

unique_ptr<NodeStatistics> VortexCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto &data = bind_data->Cast<VortexBindData>();

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

	vortex_func.init_global = [](ClientContext &context,
	                             TableFunctionInitInput &input) -> unique_ptr<GlobalTableFunctionState> {
		auto &bind = input.bind_data->Cast<VortexBindData>();
		auto state = make_uniq<VortexScanGlobalState>();

		state->filter = input.filters;
		state->projection_ids.reserve(input.projection_ids.size());
		for (auto proj_id : input.projection_ids) {
			state->projection_ids.push_back(input.column_ids[proj_id]);
		}

		state->column_ids = input.column_ids;

		// TODO(joe): do this expansion gradually in the scan to avoid a slower start.
		state->expanded_files = bind.file_list->GetAllFiles();

		state->filter_str = create_filter_expression(bind, *state);
		auto column_names = std::vector<char const *>();
		for (auto col_id : state->projection_ids) {
			assert(col_id < bind.column_names.size());
			column_names.push_back(bind.column_names[col_id].c_str());
		}
		state->projected_column_names = column_names;

		// Can ignore mutex since no other threads are running now.
		state->file_slots[0].array_stream = OpenArrayStream(bind, *state, bind.initial_file.get());
		state->next_file = 1;

		return std::move(state);
	};

	vortex_func.init_local = [](ExecutionContext &context, TableFunctionInitInput &input,
	                            GlobalTableFunctionState *global_state) -> unique_ptr<LocalTableFunctionState> {
		auto &v_global_state = global_state->Cast<VortexScanGlobalState>();

		auto thread_id = v_global_state.thread_id_counter.fetch_add(1);
		assert(thread_id < MAX_THREAD_COUNT);

		auto state = make_uniq<VortexScanLocalState>(thread_id);
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
