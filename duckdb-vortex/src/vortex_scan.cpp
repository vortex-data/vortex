#define ENABLE_DUCKDB_FFI

#include <memory>
#include <mutex>
#include <queue>
#include <regex>
#include <thread>
#include <vector>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/storage/object_cache.hpp"

#include "concurrentqueue.h"

#include "vortex.hpp"
#include "vortex_scan.hpp"
#include "vortex_common.hpp"
#include "vortex_expr.hpp"
#include "vortex_session.hpp"

using namespace duckdb;

namespace vortex {

/// Bind data for the Vortex table function that holds information about the
/// file and its schema. This data is populated during the bind phase, which
/// happens during the query planning phase.
struct BindData : public TableFunctionData {
	// Session used to caching
	shared_ptr<VortexSession> session;

	shared_ptr<MultiFileList> file_list;
	vector<LogicalType> columns_types;
	vector<string> column_names;

	// Used to read the schema during the bind phase and cached here to
	// avoid having to open the same file again during the scan phase.
	shared_ptr<FileReader> initial_file;

	// Used to create an arena for protobuf exprs, need a ptr since the bind arg is const.
	unique_ptr<google::protobuf::Arena> arena;
	vector<expr::Expr *> conjuncts;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<BindData>();
		return file_list == other.file_list && column_names == other.column_names &&
		       columns_types == other.columns_types;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<BindData>();
		result->session = session;
		result->file_list = file_list;
		result->columns_types = columns_types;
		result->column_names = column_names;
		result->initial_file = initial_file;
 		return std::move(result);
	}
};

struct ScanPartition {
	uint64_t file_idx;
	uint64_t start_row;
	uint64_t end_row;
};

/// Local state for the Vortex table function that tracks the progress of a scan
/// operation. In DuckDB's execution model, a query reading from a file can be
/// parallelized by dividing it into ranges, each handled by a different scan.
struct ScanLocalState : public LocalTableFunctionState {
	idx_t array_row_offset;
	unique_ptr<Array> currently_scanned_array;
	unique_ptr<ArrayIterator> array_iterator;
	unique_ptr<ConversionCache> conversion_cache;

	std::queue<ScanPartition> scan_partitions;

	// Thread local file.
	std::optional<idx_t> thread_local_file_idx;
};

struct ScanGlobalState : public GlobalTableFunctionState {
	std::atomic_bool finished;
	std::atomic_uint64_t cache_id;

	vector<string> expanded_files;

	optional_ptr<TableFilterSet> filter;
	// The precomputed filter string used in the query
	std::string filter_str;

	// Limited to indicate progress in `table_scan_progress`.
	std::atomic_uint32_t partitions_processed;
	std::atomic_uint32_t partitons_total;

	// Number of files which have are fully partitioned.
	std::atomic_uint32_t files_partitioned;

	// Next file to partition.
	std::atomic_uint32_t next_file_idx;

	// Multi producer, multi consumer lockfree queue.
	duckdb_moodycamel::ConcurrentQueue<ScanPartition> scan_partitions {8192};

	std::vector<shared_ptr<FileReader>> file_readers;

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
                            vector<expr::Expr *> &conjuncts) {
	if (filter == nullptr) {
		return;
	}

	for (const auto &[col_id, value] : filter->filters) {
		auto column_name = column_names[column_ids[col_id]];
		auto conj = table_expression_into_expr(arena, *value, column_name);
		conjuncts.push_back(conj);
	}
}

static void PopulateProjection(ScanGlobalState &global_state, const vector<string> &column_names,
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

static unique_ptr<FileReader> OpenFile(const std::string &filename, VortexSession &session,
                                       vector<LogicalType> &column_types, vector<string> &column_names) {
	vx_file_open_options options {
	    .uri = filename.c_str(), .property_keys = nullptr, .property_vals = nullptr, .property_len = 0};

	auto file = FileReader::Open(&options, session);
	if (!file) {
		throw IOException("Failed to open Vortex file: " + filename);
	}

	// This pointer is owned by the file.
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
static void VerifyNewFile(const BindData &bind_data, vector<LogicalType> &column_types, vector<string> &column_names) {
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

static unique_ptr<FileReader> OpenFileAndVerify(FileSystem &fs, VortexSession &session, const std::string &filename,
                                                const BindData &bind_data) {
	auto new_column_names = vector<string>();
	new_column_names.reserve(bind_data.column_names.size());

	auto new_column_types = vector<LogicalType>();
	new_column_types.reserve(bind_data.columns_types.size());

	auto file = OpenFile(EnsureFileProtocol(fs, filename), session, new_column_types, new_column_names);
	VerifyNewFile(bind_data, new_column_types, new_column_names);
	return file;
}

static bool PinFileToThread(ScanGlobalState &global_state) {
	// This is an approximation to determine whether we should switch to
	// distributing partitions of the same file across threads and does
	// not need to be exact in terms of how many threads DuckDB actually uses.
	const auto thread_count = std::thread::hardware_concurrency();
	const auto file_count = global_state.expanded_files.size();
	return (file_count - global_state.files_partitioned) > thread_count;
}

static void CreateScanPartitions(ClientContext &context, const BindData &bind, ScanGlobalState &global_state,
                                 ScanLocalState &local_state, uint64_t file_idx, FileReader &file_reader) {

	if (global_state.file_readers[file_idx]->CanPrune(global_state.filter_str.data(),
	                                                  static_cast<unsigned>(global_state.filter_str.length()))) {
		global_state.files_partitioned += 1;
		return;
	}

	const auto row_count = Try([&](auto err) { return vx_file_row_count(file_reader.file, err); });

	const auto thread_count = std::thread::hardware_concurrency();
	const auto file_count = global_state.expanded_files.size();

	// This is a multiple of the 2048 DuckDB vector size:
	//
	// Factors to consider:
	//  1. A smaller value means more work for the Vortex file reader.
	//  2. A larger value reduces the parallelism available to the scanner
	const uint64_t partition_size = 2048 * (thread_count > file_count ? 32 : 64);

	const auto partition_count = std::max(static_cast<uint64_t>(1), row_count / partition_size);
	global_state.partitons_total += partition_count;
	const bool pin_file_to_thread = PinFileToThread(global_state);

	if (pin_file_to_thread) {
		local_state.thread_local_file_idx = file_idx;
	}

	for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
		const auto scan_partition = ScanPartition {
		    .file_idx = file_idx,
		    .start_row = partition_idx * partition_size,
		    .end_row = (partition_idx + 1) == partition_count ? row_count : (partition_idx + 1) * partition_size,
		};

		if (pin_file_to_thread) {
			local_state.scan_partitions.push(scan_partition);
		} else {
			global_state.scan_partitions.enqueue(scan_partition);
		}
	}

	global_state.files_partitioned += 1;
	D_ASSERT(global_state.files_partitioned <= global_state.expanded_files.size());
}

static unique_ptr<ArrayIterator> OpenArrayIter(ScanGlobalState &global_state, shared_ptr<FileReader> &file_reader,
                                               ScanPartition row_range_partition) {
	const auto options = vx_file_scan_options {
	    .projection = global_state.projected_column_names.data(),
	    .projection_len = static_cast<unsigned>(global_state.projected_column_names.size()),
	    .filter_expression = global_state.filter_str.data(),
	    .filter_expression_len = static_cast<unsigned>(global_state.filter_str.length()),
	    .split_by_row_count = 0,
	    .row_range_start = row_range_partition.start_row,
	    .row_range_end = row_range_partition.end_row,
	};

	return make_uniq<ArrayIterator>(file_reader->Scan(&options));
}

// Assigns the next array from the array stream.
//
// Returns true if a new array was assigned, false otherwise.
static bool GetNextArray(ClientContext &context, const BindData &bind_data, ScanGlobalState &global_state,
                         ScanLocalState &local_state, DataChunk &output) {

	// Try to deque a partition off the thread local queue.
	auto try_dequeue = [&](ScanPartition &scan_partition) {
		if (local_state.scan_partitions.empty()) {
			return false;
		}

		scan_partition = local_state.scan_partitions.front();
		local_state.scan_partitions.pop();
		return true;
	};

	if (local_state.array_iterator == nullptr) {
		ScanPartition partition;

		if (bool success = (try_dequeue(partition) || global_state.scan_partitions.try_dequeue(partition)); !success) {

			// Check whether all partitions have been processed.
			if (global_state.files_partitioned == global_state.expanded_files.size()) {

				// A new partition might have been created after the first pop. Therefore,
				// one more pop is necessary to ensure no more partitions are left to process.
				if (success = global_state.scan_partitions.try_dequeue(partition); !success) {
					global_state.finished = true;
					return false;
				}
			}

			if (!success) {
				return false;
			}
		}

		// Layout readers are safe to share across threads for reading. Further, they
		// are created before pushing partitions of the corresponing files into a queue.
		auto file_reader = global_state.file_readers[partition.file_idx];
		local_state.array_iterator = OpenArrayIter(global_state, file_reader, partition);
	}

	local_state.currently_scanned_array = local_state.array_iterator->NextArray();
	local_state.array_row_offset = 0;

	if (local_state.currently_scanned_array == nullptr) {
		local_state.array_iterator = nullptr;
		global_state.partitions_processed += 1;

		return false;
	}

	return true;
}

static void VortexScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<BindData>();
	auto &global_state = data.global_state->Cast<ScanGlobalState>();
	auto &local_state = data.local_state->Cast<ScanLocalState>();

	if (local_state.currently_scanned_array == nullptr) {
		while (!GetNextArray(context, bind_data, global_state, local_state, output)) {
			if (global_state.finished) {
				output.Reset();
				output.SetCardinality(0);
				return;
			}

			// Free file readers when owned by the thread.
			if (local_state.scan_partitions.empty() && local_state.thread_local_file_idx.has_value()) {
				global_state.file_readers[local_state.thread_local_file_idx.value()] = nullptr;
				local_state.thread_local_file_idx.reset();
			}

			// Create new scan partitions in case the queue is empty.
			if (auto file_idx = global_state.next_file_idx.fetch_add(1);
			    file_idx < global_state.expanded_files.size()) {
				if (file_idx == 0) {
					global_state.file_readers[0] = bind_data.initial_file;
				} else {
					auto file_name = global_state.expanded_files[file_idx];
					global_state.file_readers[file_idx] =
					    OpenFileAndVerify(FileSystem::GetFileSystem(context), *bind_data.session, file_name, bind_data);
				}

				CreateScanPartitions(context, bind_data, global_state, local_state, file_idx,
				                     *global_state.file_readers[file_idx]);
			}
		}
	}

	if (local_state.conversion_cache == nullptr) {
		local_state.conversion_cache = make_uniq<ConversionCache>(global_state.cache_id++);
	}

	local_state.array_row_offset = local_state.currently_scanned_array->ToDuckDBVector(
	    local_state.array_row_offset, reinterpret_cast<duckdb_data_chunk>(&output), local_state.conversion_cache.get());

	if (local_state.array_row_offset == 0) {
		local_state.currently_scanned_array = nullptr;
		local_state.conversion_cache = nullptr;
	}
}

/// The bind function (for the Vortex table function) is called during query
/// planning. The bind phase happens once per query and allows DuckDB to know
/// the schema of the data before execution begins. This enables optimizations
/// like projection pushdown and predicate pushdown.
static unique_ptr<FunctionData> VortexBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &column_types, vector<string> &column_names) {
	auto result = make_uniq<BindData>();
	result->arena = make_uniq<google::protobuf::Arena>();

	const static string VortexExtensionKey = std::string("vortex_extension:vortex_session");
	auto session = ObjectCache::GetObjectCache(context).Get<VortexSession>(VortexExtensionKey);
	if (session == nullptr) {
		ObjectCache::GetObjectCache(context).Put(VortexExtensionKey, make_shared_ptr<VortexSession>());
		session = ObjectCache::GetObjectCache(context).Get<VortexSession>(VortexExtensionKey);
	}

	result->session = session;

	auto file_glob_strings = duckdb::vector<string> {input.inputs[0].GetValue<string>()};
	auto file_glob = duckdb::vector<OpenFileInfo>(file_glob_strings.begin(), file_glob_strings.end());
	result->file_list = make_shared_ptr<GlobMultiFileList>(context, file_glob, FileGlobOptions::DISALLOW_EMPTY);

	// Open the first file to extract the schema.
	auto filename = EnsureFileProtocol(FileSystem::GetFileSystem(context), result->file_list->GetFirstFile().path);
	result->initial_file = OpenFile(filename, *result->session, column_types, column_names);

	result->column_names = column_names;
	result->columns_types = column_types;

	return std::move(result);
}

unique_ptr<NodeStatistics> VortexCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto &data = bind_data->Cast<BindData>();

	auto row_count = data.initial_file->FileRowCount();
	if (data.file_list->GetTotalFileCount() == 1) {
		return make_uniq<NodeStatistics>(row_count, row_count);
	} else {
		return make_uniq<NodeStatistics>(row_count * data.file_list->GetTotalFileCount());
	}
}

// Removes all filter expressions (from `filters`) which can be pushed down.
void PushdownComplexFilter(ClientContext &context, LogicalGet &get, FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
	if (filters.empty()) {
		return;
	}

	auto &bind = bind_data->Cast<BindData>();
	bind.conjuncts.reserve(filters.size());

	for (auto &filter : filters) {
		if (auto expr = expression_into_vortex_expr(*bind.arena, *filter); expr != nullptr) {
			bind.conjuncts.push_back(expr);
		}
	}
}

void RegisterScanFunction(DatabaseInstance &instance) {

	TableFunction vortex_scan("read_vortex", {LogicalType::VARCHAR}, VortexScanFunction, VortexBind);

	vortex_scan.init_global = [](ClientContext &context,
	                             TableFunctionInitInput &input) -> unique_ptr<GlobalTableFunctionState> {
		auto &bind = input.bind_data->CastNoConst<BindData>();
		auto global_state = make_uniq<ScanGlobalState>();

		// TODO(joe): do this expansion gradually in the scan to avoid a slower start.
		auto file_infos = bind.file_list->GetAllFiles();
		global_state->expanded_files.reserve(file_infos.size());
		for (const auto &file_info : file_infos) {
			global_state->expanded_files.push_back(file_info.path);
		}
		global_state->filter = input.filters;
		global_state->column_ids = input.column_ids;

		PopulateProjection(*global_state, bind.column_names, input);

		// Most expressions are extracted from `PushdownComplexFilter`, the final filters come from `input.filters`.
		CreateFilterExpression(*bind.arena, bind.column_names, input.filters, input.column_ids, bind.conjuncts);
		if (auto exprs = flatten_exprs(*bind.arena, bind.conjuncts); exprs != nullptr) {
			global_state->filter_str = exprs->SerializeAsString();
		}

		// Resizing the empty vector default constructs std::shared pointers at all indices with nullptr.
		global_state->file_readers.resize(global_state->expanded_files.size());

		bind.arena->Reset();
		return std::move(global_state);
	};

	vortex_scan.init_local = [](ExecutionContext &context, TableFunctionInitInput &input,
	                            GlobalTableFunctionState *global_state) -> unique_ptr<LocalTableFunctionState> {
		return make_uniq<ScanLocalState>();
	};

	vortex_scan.table_scan_progress = [](ClientContext &context, const FunctionData *bind_data,
	                                     const GlobalTableFunctionState *global_state) -> double {
		auto &gstate = global_state->Cast<ScanGlobalState>();
		return 100.0 * (static_cast<double>(gstate.partitions_processed) / static_cast<double>(gstate.partitons_total));
	};

	vortex_scan.pushdown_complex_filter = PushdownComplexFilter;
	vortex_scan.projection_pushdown = true;
	vortex_scan.cardinality = VortexCardinality;
	vortex_scan.filter_pushdown = true;
	vortex_scan.filter_prune = true;

	ExtensionUtil::RegisterFunction(instance, vortex_scan);
}

} // namespace vortex
