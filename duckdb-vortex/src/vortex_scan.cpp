#include "vortex_scan.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/common/file_system.hpp"
#include "vortex.hpp"
#include "vortex_extension.hpp"
#include "vortex_layout_reader.hpp"

#include <memory>
#include <mutex>
#include <regex>

#include "vortex_common.hpp"
#include "expr/expr.hpp"

namespace duckdb {

// This is a multiple of the 2048 DuckDB vector size, it needs tuning
// This has a few factor effecting it:
//  1. A smaller value means for work for the vortex file reader.
//  2. A larger value reduces the parallelism available to the scanner
constexpr uint64_t PARTITION_SIZE = 2048 * 32;

/// Bind data for the Vortex table function that holds information about the
/// file and its schema. This data is populated during the bind phase, which
/// happens during the query planning phase.
struct VortexBindData : public TableFunctionData {
	shared_ptr<MultiFileList> file_list;
	vector<LogicalType> columns_types;
	vector<string> column_names;

	// Used to read the schema during the bind phase and cached here to
	// avoid having to open the same file again during the scan phase.
	unique_ptr<VortexFileReader> initial_file;

	// Used to create an arena for protobuf exprs, need a ptr since the bind arg is const.
	unique_ptr<google::protobuf::Arena> arena;
	vector<vortex::expr::Expr *> conjuncts;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<VortexBindData>();
		return file_list == other.file_list && column_names == other.column_names &&
		       columns_types == other.columns_types;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<VortexBindData>();
		result->file_list = file_list;
		result->columns_types = columns_types;
		result->column_names = column_names;
		return std::move(result);
	}
};

/// Local state for the Vortex table function that tracks the progress of a scan
/// operation. In DuckDB's execution model, a query reading from a file can be
/// parallelized by dividing it into ranges, each handled by a different scan.
struct VortexScanLocalState : public LocalTableFunctionState {
	idx_t current_row;
	unique_ptr<VortexArray> array;
	unique_ptr<VortexArrayStream> stream;
	unique_ptr<VortexConversionCache> cache;
};

struct VortexScanPartition {
	uint64_t file_idx;
	uint64_t start_row;
	uint64_t end_row;
};

struct VortexScanGlobalState : public GlobalTableFunctionState {
	std::atomic_bool finished;
	std::atomic_uint64_t cache_id;

	vector<string> expanded_files;

	optional_ptr<TableFilterSet> filter;
	// The precomputed filter string used in the query
	std::string filter_str;

	std::vector<VortexScanPartition> scan_partitions;
	std::atomic_uint32_t next_partition;

	std::vector<std::shared_ptr<VortexLayoutReader>> layout_readers;
	std::vector<std::mutex> layout_mutexes;

	// The column idx that must be returned by the scan.
	vector<idx_t> column_ids;
	vector<idx_t> projection_ids;
	// The precomputed column names used in the query.
	std::vector<char const *> projected_column_names;

	// This is the max number threads that the extension might use.
	idx_t MaxThreads() const override {
		constexpr uint32_t MAX_THREAD_COUNT = 192;
		return MAX_THREAD_COUNT;
	}
};

// Use to create vortex expressions from `TableFilterSet` filter.
void CreateFilterExpression(google::protobuf::Arena &arena, vector<std::string> column_names,
                            optional_ptr<TableFilterSet> filter, vector<idx_t> column_ids,
                            vector<vortex::expr::Expr *> &conjuncts) {
	if (filter == nullptr) {
		return;
	}

	for (const auto &[col_id, value] : filter->filters) {
		auto column_name = column_names[column_ids[col_id]];
		auto conj = table_expression_into_expr(arena, *value, column_name);
		conjuncts.push_back(conj);
	}
}

static void PopulateProjection(VortexScanGlobalState &global_state, const vector<string> &column_names,
                               TableFunctionInitInput &input) {
	global_state.projection_ids.reserve(input.projection_ids.size());
	for (auto proj_id : input.projection_ids) {
		global_state.projection_ids.push_back(input.column_ids[proj_id]);
	}

	global_state.projected_column_names.reserve(input.projection_ids.size());
	for (auto column_id : global_state.projection_ids) {
		assert(column_id < column_names.size());
		global_state.projected_column_names.push_back(column_names[column_id].c_str());
	}
}

/// Extracts schema information from a Vortex file's data type.
static void ExtractVortexSchema(DType &file_dtype, vector<LogicalType> &column_types, vector<string> &column_names) {
	uint32_t field_count = vx_dtype_field_count(file_dtype.dtype);
	for (uint32_t idx = 0; idx < field_count; idx++) {
		char name_buffer[512];
		int name_len = 0;

		vx_dtype_field_name(file_dtype.dtype, idx, name_buffer, &name_len);
		std::string field_name(name_buffer, name_len);

		vx_dtype *field_dtype = vx_dtype_field_dtype(file_dtype.dtype, idx);
		auto duckdb_type = Try([&](auto err) { return vx_dtype_to_duckdb_logical_type(field_dtype, err); });

		column_names.push_back(field_name);
		column_types.push_back(LogicalType(*reinterpret_cast<LogicalType *>(duckdb_type)));
		vx_dtype_free(field_dtype);
		duckdb_destroy_logical_type(&duckdb_type);
	}
}

std::string EnsureFileProtocol(FileSystem &fs, const std::string &path) {
	// If the path is a URL then don't change it, otherwise try to make the path an absolute path
	static const std::regex schema_prefix = std::regex("^[^/]*:\\/\\/.*$");
	if (std::regex_match(path, schema_prefix)) {
		return path;
	}

	const std::string prefix = "file://";
	if (fs.IsPathAbsolute(path)) {
		return prefix + path;
	}

	const auto absolute_path = fs.JoinPath(fs.GetWorkingDirectory(), path);
	return prefix + absolute_path;
}

static unique_ptr<VortexFileReader> OpenFile(const std::string &filename, vector<LogicalType> &column_types,
                                             vector<string> &column_names) {
	vx_file_open_options options {
	    .uri = filename.c_str(), .property_keys = nullptr, .property_vals = nullptr, .property_len = 0};

	auto file = VortexFileReader::Open(&options);
	if (!file) {
		throw IOException("Failed to open Vortex file: " + filename);
	}

	// This Ptr is owned by the file
	auto file_dtype = file->DType();
	if (vx_dtype_get(file_dtype.dtype) != DTYPE_STRUCT) {
		vx_file_reader_free(file->file);
		throw FatalException("Vortex file does not contain a struct array as a top-level dtype");
	}

	ExtractVortexSchema(file_dtype, column_types, column_names);

	return file;
}

// Verifies that a new Vortex file's schema matches the expected schema from the bind phase.
//
// This function ensures schema consistency across all the files in a multi-file query.
// It compares the column types and names extracted from a new file against the schema
// obtained from the first file (stored in bind_data).
static void VerifyNewFile(const VortexBindData &bind_data, vector<LogicalType> &column_types,
                          vector<string> &column_names) {
	if (column_types.size() != bind_data.columns_types.size() || column_names != bind_data.column_names) {
		throw FatalException("Vortex file does not contain the same number of columns as the first");
	}

	for (size_t idx = 0; idx < bind_data.columns_types.size(); ++idx) {
		if (bind_data.column_names[idx] != column_names[idx]) {
			throw FatalException("Vortex file contains a column with a different name to the first");
		}
		if (bind_data.columns_types[idx] != column_types[idx]) {
			throw FatalException("Vortex file contains a column with a different type to the first");
		}
	}
}

static unique_ptr<VortexFileReader> OpenFileAndVerify(FileSystem &fs, const std::string &filename,
                                                      const VortexBindData &bind_data) {
	auto new_column_names = vector<string>();
	new_column_names.reserve(bind_data.column_names.size());

	auto new_column_types = vector<LogicalType>();
	new_column_types.reserve(bind_data.columns_types.size());

	auto file = OpenFile(EnsureFileProtocol(fs, filename), new_column_types, new_column_names);
	VerifyNewFile(bind_data, new_column_types, new_column_names);
	return file;
}

static void CreateScanPartitions(ClientContext &context, const VortexBindData &bind,
                                 VortexScanGlobalState &global_state) {
	uint64_t file_idx = 0;
	for (const auto &file_name : global_state.expanded_files) {
		auto file_reader = OpenFileAndVerify(FileSystem::GetFileSystem(context), file_name, bind);

		const uint64_t row_count = Try([&](auto err) { return vx_file_row_count(file_reader->file, err); });
		const auto partition_count = std::max(static_cast<uint64_t>(1), row_count / PARTITION_SIZE);

		for (uint64_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
			global_state.scan_partitions.push_back(VortexScanPartition {
			    .file_idx = file_idx,
			    .start_row = partition_idx * PARTITION_SIZE,
			    .end_row = (partition_idx + 1) * PARTITION_SIZE,
			});
		}

		global_state.scan_partitions.back().end_row = row_count;

		++file_idx;
	}
}

static unique_ptr<VortexArrayStream> OpenArrayStream(VortexScanGlobalState &global_state,
                                                     std::shared_ptr<VortexLayoutReader> &layout_reader,
                                                     VortexScanPartition row_range_partition) {
	const auto options = vx_file_scan_options {
	    .projection = global_state.projected_column_names.data(),
	    .projection_len = static_cast<int>(global_state.projected_column_names.size()),
	    .filter_expression = global_state.filter_str.data(),
	    .filter_expression_len = static_cast<int>(global_state.filter_str.length()),
	    .split_by_row_count = 0,
	    .row_range_start = row_range_partition.start_row,
	    .row_range_end = row_range_partition.end_row,
	};

	return make_uniq<VortexArrayStream>(layout_reader->Scan(&options));
}

// Assigns the next array from the array stream.
//
// Returns true if a new array was assigned. Returns false otherwise.
static bool GetNextArray(ClientContext &context, const VortexBindData &bind_data, VortexScanGlobalState &global_state,
                         VortexScanLocalState &local_state, DataChunk &output) {

	auto partition_idx = global_state.next_partition.fetch_add(1);

	// No more partitions to read.
	if (partition_idx >= global_state.scan_partitions.size()) {
		global_state.finished = true;
		return false;
	}

	if (local_state.array == nullptr) {
		auto partition = global_state.scan_partitions[partition_idx];

		std::shared_ptr<VortexLayoutReader> layout_reader = [&] {
			std::lock_guard<std::mutex> lock(global_state.layout_mutexes[partition.file_idx]);

			if (global_state.layout_readers[partition.file_idx]) {
				return global_state.layout_readers[partition.file_idx];
			} else {
				auto file_name = global_state.expanded_files[partition.file_idx];
				auto vortex_file = OpenFileAndVerify(FileSystem::GetFileSystem(context), file_name, bind_data);
				global_state.layout_readers[partition.file_idx] = VortexLayoutReader::CreateFromFile(vortex_file.get());
				return global_state.layout_readers[partition.file_idx];
			}
		}();

		local_state.stream = OpenArrayStream(global_state, layout_reader, partition);
	}

	local_state.array = local_state.stream->NextArray();

	// Reset row offset for the array.
	local_state.current_row = 0;

	// If the stream is empty, mark it as read by returning false.
	if (local_state.array == nullptr) {
		local_state.stream = nullptr;
		return false;
	}

	return true;
}

static void VortexScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<VortexBindData>();
	auto &global_state = data.global_state->Cast<VortexScanGlobalState>();
	auto &local_state = data.local_state->Cast<VortexScanLocalState>();

	if (global_state.finished && local_state.stream == nullptr) {
		// Return an empty data chunk if all partitions have been processed.
		output.Reset();
		output.SetCardinality(0);
		return;
	}

	if (local_state.array == nullptr) {
		while (!GetNextArray(context, bind_data, global_state, local_state, output)) {
			return;
		}
	}

	if (local_state.cache == nullptr) {
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
	result->arena = make_uniq<google::protobuf::Arena>();

	auto file_glob = duckdb::vector<string> {input.inputs[0].GetValue<string>()};
	result->file_list = make_shared_ptr<GlobMultiFileList>(context, file_glob, FileGlobOptions::DISALLOW_EMPTY);

	// Open the first file to extract the schema.
	auto filename = EnsureFileProtocol(FileSystem::GetFileSystem(context), result->file_list->GetFirstFile());
	result->initial_file = OpenFile(filename, column_types, column_names);

	result->column_names = column_names;
	result->columns_types = column_types;

	return std::move(result);
}

unique_ptr<NodeStatistics> VortexCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto &data = bind_data->Cast<VortexBindData>();
	return make_uniq<NodeStatistics>(data.column_names.size(), data.column_names.size());
}

// Removes all filter expressions (from `filters`) which can be pushed down.
void PushdownComplexFilter(ClientContext &context, LogicalGet &get, FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
	if (filters.empty()) {
		return;
	}

	auto &bind = bind_data->Cast<VortexBindData>();
	bind.conjuncts.reserve(filters.size());

	for (auto &filter : filters) {
		if (auto expr = expression_into_vortex_expr(*bind.arena, *filter); expr != nullptr) {
			bind.conjuncts.push_back(expr);
		}
	}
}

void RegisterVortexScanFunction(DatabaseInstance &instance) {

	TableFunction vortex_scan("read_vortex", {LogicalType::VARCHAR}, VortexScanFunction, VortexBind);

	vortex_scan.init_global = [](ClientContext &context,
	                             TableFunctionInitInput &input) -> unique_ptr<GlobalTableFunctionState> {
		auto &bind = input.bind_data->CastNoConst<VortexBindData>();
		auto global_state = make_uniq<VortexScanGlobalState>();

		// TODO(joe): do this expansion gradually in the scan to avoid a slower start.
		global_state->expanded_files = bind.file_list->GetAllFiles();
		global_state->filter = input.filters;
		global_state->column_ids = input.column_ids;

		PopulateProjection(*global_state, bind.column_names, input);

		// Most expressions are extracted from `PushdownComplexFilter`, the final filters come from `input.filters`.
		CreateFilterExpression(*bind.arena, bind.column_names, input.filters, input.column_ids, bind.conjuncts);
		if (auto exprs = flatten_exprs(*bind.arena, bind.conjuncts); exprs != nullptr) {
			global_state->filter_str = exprs->SerializeAsString();
		}

		// Resizing the empty vector default constructs std::shared pointers at all indices with nullptr.
		global_state->layout_readers.resize(global_state->expanded_files.size());
		global_state->layout_mutexes = std::vector<std::mutex>(global_state->expanded_files.size());

		CreateScanPartitions(context, bind, *global_state);

		// Retrieve the first layout reader from the initial file which is already open.
		global_state->layout_readers[0] = VortexLayoutReader::CreateFromFile(bind.initial_file.get());

		bind.arena->Reset();
		return std::move(global_state);
	};

	vortex_scan.init_local = [](ExecutionContext &context, TableFunctionInitInput &input,
	                            GlobalTableFunctionState *global_state) -> unique_ptr<LocalTableFunctionState> {
		return make_uniq<VortexScanLocalState>();
	};

	vortex_scan.table_scan_progress = [](ClientContext &context, const FunctionData *bind_data,
	                                     const GlobalTableFunctionState *global_state) -> double {
		auto &gstate = global_state->Cast<VortexScanGlobalState>();

		return 100.0 *
		       (static_cast<double>(gstate.next_partition.load()) / static_cast<double>(gstate.scan_partitions.size()));
	};

	vortex_scan.pushdown_complex_filter = PushdownComplexFilter;
	vortex_scan.projection_pushdown = true;
	vortex_scan.cardinality = VortexCardinality;
	vortex_scan.filter_pushdown = true;
	vortex_scan.filter_prune = true;

	ExtensionUtil::RegisterFunction(instance, vortex_scan);
}

} // namespace duckdb
