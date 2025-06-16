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
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"

using namespace duckdb;

namespace vortex {

static constexpr column_t _COLUMN_IDENTIFIER_FILE_ROW_NUMBER = UINT64_C(9223372036854775809);
static constexpr column_t _COLUMN_IDENTIFIER_FILE_INDEX = UINT64_C(9223372036854775810);

/// Bind data for the Vortex table function that holds information about the
/// file and its schema. This data is populated during the bind phase, which
/// happens during the query planning phase.
struct ScanBindData : public TableFunctionData {
	// Session used to caching
	shared_ptr<VortexSession> session;

	shared_ptr<MultiFileList> file_list;
	vector<LogicalType> columns_types;
	vector<string> column_names;
	map<column_t, string> virtual_col;

	// Used to read the schema during the bind phase and cached here to
	// avoid having to open the same file again during the scan phase.
	shared_ptr<VortexFile> initial_file;

	// Used to create an arena for protobuf exprs, need a ptr since the bind arg is const.
	unique_ptr<google::protobuf::Arena> arena;
	vector<expr::Expr *> conjuncts;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ScanBindData>();
		return file_list == other.file_list && column_names == other.column_names &&
		       columns_types == other.columns_types;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ScanBindData>();
		result->arena = make_uniq<google::protobuf::Arena>();
		result->session = session;
		result->file_list = file_list;
		result->columns_types = columns_types;
		result->column_names = column_names;
		result->initial_file = initial_file;
		result->virtual_col = virtual_col;
		return std::move(result);
	}

	std::string &ColumnName(column_t col) {
		if (col < column_names.size()) {
			return column_names[col];
		}

		return virtual_col[col];
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
	bool finished;

	unique_ptr<ArrayExporter> array_exporter;

	std::queue<ScanPartition> scan_partitions;

	// Thread local file.
	std::optional<idx_t> thread_local_file_idx;
};

struct ScanGlobalState : public GlobalTableFunctionState {
	std::atomic_uint64_t cache_id;

	vector<string> expanded_files;

	optional_ptr<TableFilterSet> filter;
	// The precomputed filter string used in the query
	std::string static_filter_str;
	// Any dynamic filter contained in the query
	map<string, shared_ptr<DynamicFilterData>> dynamic_filters;
	// Static filter expression, owned by the arena
	expr::Expr *static_filter_expr;

	// Limited to indicate progress in `table_scan_progress`.
	std::atomic_uint32_t partitions_processed;
	std::atomic_uint32_t partitons_total;

	// Number of files which have are fully partitioned.
	std::atomic_uint32_t files_partitioned;

	// Next file to partition.
	std::atomic_uint32_t next_file_idx;

	// Multi producer, multi consumer lockfree queue.
	duckdb_moodycamel::ConcurrentQueue<ScanPartition> scan_partitions {8192};

	std::vector<shared_ptr<VortexFile>> files;

	// The column idx that must be returned by the scan.
	vector<idx_t> column_ids;
	vector<idx_t> projection_ids;
	// The precomputed column names used in the query.
	std::string projection;

	// This is the max number threads that the extension might use.
	idx_t MaxThreads() const override {
		constexpr uint32_t MAX_THREAD_COUNT = 192;
		return MAX_THREAD_COUNT;
	}

	// Return the conjunction of the static and dynamic filters, if either exist.
	// The dynamic filter can be updated and so we recompute the filter if there is an
	// active dyn filter.
	// TODO(joe): cache the dyn filter expr if the dynamic filters have not changed.
	std::string filter_expression_string(google::protobuf::Arena &arena) {
		if (dynamic_filters.empty()) {
			return static_filter_str;
		}
		vector<expr::Expr *> conjs;
		for (auto &[col_name, filter] : dynamic_filters) {
			auto g = lock_guard(filter->lock);
			if (!filter->initialized) {
				continue;
			}
			auto conj = table_expression_into_expr(arena, *filter->filter, col_name);
			if (conj) {
				conjs.push_back(conj);
			}
		}
		if (conjs.empty()) {
			return static_filter_expr->SerializeAsString();
		}
		auto dynamic_expr = flatten_exprs(arena, conjs);
		auto expr = arena.Create<expr::Expr>(&arena);
		expr->set_id(BINARY_ID);
		expr->mutable_kind()->set_binary_op(expr::Kind::And);
		expr->add_children()->Swap(dynamic_expr);
		expr->add_children()->CopyFrom(*static_filter_expr);
		return expr->SerializeAsString();
	}
};

// Use to create vortex expressions from `TableFilterSet` filter.
void ExtractFilterExpression(google::protobuf::Arena &arena, ScanBindData &data,
                             optional_ptr<TableFilterSet> filter_set, vector<idx_t> column_ids,
                             vector<expr::Expr *> &conjuncts, map<string, shared_ptr<DynamicFilterData>> &dyn_filters) {
	if (filter_set == nullptr) {
		return;
	}

	for (const auto &[col_id, value] : filter_set->filters) {
		auto column_name = data.ColumnName(column_ids[col_id]);

		// Extract the optional dynamic filter, this seems like the only way that
		// duckdb will use dynamic filters.
		if (value->filter_type == TableFilterType::OPTIONAL_FILTER) {
			auto &opt_filter = value->Cast<OptionalFilter>().child_filter;
			if (opt_filter->filter_type == TableFilterType::DYNAMIC_FILTER) {
				dyn_filters.emplace(column_name, opt_filter->Cast<DynamicFilter>().filter_data);
				continue;
			}
		}
		auto conj = table_expression_into_expr(arena, *value, column_name);
		conjuncts.push_back(conj);
	}
}

static void PopulateProjection(ScanBindData &bind_data, ScanGlobalState &global_state, TableFunctionInitInput &input) {
	global_state.projection_ids.reserve(input.projection_ids.size());
	for (auto proj_id : input.projection_ids) {
		global_state.projection_ids.push_back(input.column_ids[proj_id]);
	}

	auto vec = duckdb::vector<std::string>();
	for (auto column_id : global_state.projection_ids) {
		vec.push_back(bind_data.ColumnName(column_id));
	}
	auto expr = pack_projection_columns(*bind_data.arena, vec);
	global_state.projection = expr->SerializeAsString();
}

/// Extracts schema information from a Vortex file's data type.
static void ExtractVortexSchema(DType &file_dtype, vector<LogicalType> &column_types, vector<string> &column_names) {
	auto struct_dtype = vx_dtype_struct_dtype(file_dtype.dtype);
	uint32_t field_count = vx_struct_fields_nfields(struct_dtype);
	for (uint32_t idx = 0; idx < field_count; idx++) {
		auto vx_field_name = vx_struct_fields_field_name(struct_dtype, idx);
		std::string field_name(vx_string_ptr(vx_field_name), vx_string_len(vx_field_name));

		const vx_dtype *field_dtype = vx_struct_fields_field_dtype(struct_dtype, idx);
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

static unique_ptr<VortexFile> OpenFile(const std::string &filename, VortexSession &session,
                                       vector<LogicalType> &column_types, vector<string> &column_names) {
	vx_file_open_options options {
	    .uri = filename.c_str(), .property_keys = nullptr, .property_vals = nullptr, .property_len = 0};

	auto file = VortexFile::Open(&options, session);
	if (!file) {
		throw IOException("Failed to open Vortex file: " + filename);
	}

	// This pointer is owned by the file.
	auto file_dtype = file->DType();
	if (vx_dtype_get_variant(file_dtype.dtype) != DTYPE_STRUCT) {
		vx_file_free(file->file);
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
static void VerifyNewFile(const ScanBindData &bind_data, vector<LogicalType> &column_types,
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

static unique_ptr<VortexFile> OpenFileAndVerify(FileSystem &fs, VortexSession &session, const std::string &filename,
                                                const ScanBindData &bind_data) {
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

static void CreateScanPartitions(ClientContext &context, const ScanBindData &bind, ScanGlobalState &global_state,
                                 ScanLocalState &local_state, uint64_t file_idx, VortexFile &file) {
	auto filter_str = global_state.filter_expression_string(*bind.arena);
	if (global_state.files[file_idx]->CanPrune(filter_str.data(), static_cast<unsigned>(filter_str.length()),
	                                           file_idx)) {
		global_state.files_partitioned += 1;
		return;
	}

	const auto row_count = vx_file_row_count(file.file);

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

static unique_ptr<ArrayIterator> OpenArrayIter(const ScanBindData &bind, ScanGlobalState &global_state,
                                               shared_ptr<VortexFile> &file, ScanPartition row_range_partition) {
	auto filter_str = global_state.filter_expression_string(*bind.arena);

	const auto options =
	    vx_file_scan_options {.projection_expression = global_state.projection.data(),
	                          .projection_expr_len = static_cast<unsigned>(global_state.projection.length()),
	                          .filter_expression = filter_str.data(),
	                          .filter_expression_len = static_cast<unsigned>(filter_str.length()),
	                          .split_by_row_count = 0,
	                          .row_range_start = row_range_partition.start_row,
	                          .row_range_end = row_range_partition.end_row,
	                          .file_index = row_range_partition.file_idx};

	return make_uniq<ArrayIterator>(file->Scan(&options));
}

// Assigns the next array exporter.
//
// Returns true if a new exporter was assigned, false otherwise.
static bool GetNextExporter(ClientContext &context, const ScanBindData &bind_data, ScanGlobalState &global_state,
                            ScanLocalState &local_state) {

	// Try to deque a partition off the thread local queue.
	auto try_dequeue = [&](ScanPartition &scan_partition) {
		if (local_state.scan_partitions.empty()) {
			return false;
		}

		scan_partition = local_state.scan_partitions.front();
		local_state.scan_partitions.pop();
		return true;
	};

	if (local_state.array_exporter == nullptr) {
		ScanPartition partition;

		if (bool success = (try_dequeue(partition) || global_state.scan_partitions.try_dequeue(partition)); !success) {

			// Check whether all partitions have been processed.
			if (global_state.files_partitioned == global_state.expanded_files.size()) {

				// A new partition might have been created after the first pop. Therefore,
				// one more pop is necessary to ensure no more partitions are left to process.
				if (success = global_state.scan_partitions.try_dequeue(partition); !success) {
					local_state.finished = true;
					return false;
				}
			}

			if (!success) {
				return false;
			}
		}

		// Layout readers are safe to share across threads for reading. Further, they
		// are created before pushing partitions of the corresponing files into a queue.
		auto file = global_state.files[partition.file_idx];
		auto array_iter = OpenArrayIter(bind_data, global_state, file, partition);
		local_state.array_exporter = ArrayExporter::FromArrayIterator(std::move(array_iter));
	}

	return true;
}

static void VortexScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<ScanBindData>();
	auto &global_state = data.global_state->Cast<ScanGlobalState>();
	auto &local_state = data.local_state->Cast<ScanLocalState>();

	while (true) {
		if (local_state.array_exporter != nullptr) {
			if (local_state.array_exporter->ExportNext(reinterpret_cast<duckdb_data_chunk>(&output))) {
				// Successfully exported a chunk
				return;
			} else {
				// Otherwise, reset the exporter and try the next one.
				global_state.partitions_processed += 1;
				local_state.array_exporter = nullptr;
			}
		}

		if (!local_state.finished) {
			// Try to get the next exporter, if we fail, make progress on partitions and then loop.
			if (!GetNextExporter(context, bind_data, global_state, local_state)) {
				// Free file readers when owned by the thread.
				if (local_state.scan_partitions.empty() && local_state.thread_local_file_idx.has_value()) {
					global_state.files[local_state.thread_local_file_idx.value()] = nullptr;
					local_state.thread_local_file_idx.reset();
				}

				// Create new scan partitions in case the queue is empty.
				if (auto file_idx = global_state.next_file_idx.fetch_add(1);
				    file_idx < global_state.expanded_files.size()) {
					if (file_idx == 0) {
						global_state.files[0] = bind_data.initial_file;
					} else {
						auto file_name = global_state.expanded_files[file_idx];
						global_state.files[file_idx] = OpenFileAndVerify(FileSystem::GetFileSystem(context),
						                                                 *bind_data.session, file_name, bind_data);
					}

					CreateScanPartitions(context, bind_data, global_state, local_state, file_idx,
					                     *global_state.files[file_idx]);
				}
			}
			continue;
		}

		// Otherwise, we're truly done.
		output.Reset();
		return;
	}
}

/// The bind function (for the Vortex table function) is called during query
/// planning. The bind phase happens once per query and allows DuckDB to know
/// the schema of the data before execution begins. This enables optimizations
/// like projection pushdown and predicate pushdown.
static unique_ptr<FunctionData> VortexBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &column_types, vector<string> &column_names) {
	auto result = make_uniq<ScanBindData>();
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
	auto &data = bind_data->Cast<ScanBindData>();

	auto row_count = data.initial_file->RowCount();
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

	auto &bind = bind_data->Cast<ScanBindData>();
	bind.conjuncts.reserve(filters.size());

	// Delete filters here so they are not given to used the create global state callback.
	for (auto iter = filters.begin(); iter != filters.end();) {
		auto expr = expression_into_vortex_expr(*bind.arena, *iter->get());
		if (expr != nullptr) {
			bind.conjuncts.push_back(expr);

			iter = filters.erase(iter);
		} else {
			++iter;
		}
	}
}

void RegisterScanFunction(DatabaseInstance &instance) {

	TableFunction vortex_scan("read_vortex", {LogicalType::VARCHAR}, VortexScanFunction, VortexBind);

	vortex_scan.init_global = [](ClientContext &context,
	                             TableFunctionInitInput &input) -> unique_ptr<GlobalTableFunctionState> {
		auto &bind = input.bind_data->CastNoConst<ScanBindData>();
		auto global_state = make_uniq<ScanGlobalState>();

		// TODO(joe): do this expansion gradually in the scan to avoid a slower start.
		auto file_infos = bind.file_list->GetAllFiles();
		global_state->expanded_files.reserve(file_infos.size());
		for (const auto &file_info : file_infos) {
			global_state->expanded_files.push_back(file_info.path);
		}
		global_state->filter = input.filters;
		global_state->column_ids = input.column_ids;

		PopulateProjection(bind, *global_state, input);

		// Most expressions are extracted from `PushdownComplexFilter`, the final filters come from `input.filters`.
		ExtractFilterExpression(*bind.arena, bind, input.filters, input.column_ids, bind.conjuncts,
		                        global_state->dynamic_filters);

		// Create the static filter expression
		global_state->static_filter_expr = flatten_exprs(*bind.arena, bind.conjuncts);
		if (global_state->static_filter_expr != nullptr) {
			global_state->static_filter_str = global_state->static_filter_expr->SerializeAsString();
		}

		// Resizing the empty vector default constructs std::shared pointers at all indices with nullptr.
		global_state->files.resize(global_state->expanded_files.size());

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
	vortex_scan.late_materialization = true;

	vortex_scan.get_row_id_columns = [](ClientContext &context,
	                                    optional_ptr<FunctionData> bind_data) -> vector<column_t> {
		vector<column_t> result;
		result.emplace_back(_COLUMN_IDENTIFIER_FILE_ROW_NUMBER);
		result.emplace_back(_COLUMN_IDENTIFIER_FILE_INDEX);
		return result;
	};
	vortex_scan.get_virtual_columns = [](ClientContext &context,
	                                     optional_ptr<FunctionData> bind_data) -> virtual_column_map_t {
		auto &scan_bind_data = bind_data->Cast<ScanBindData>();
		virtual_column_map_t result;

		result.insert(
		    make_pair(_COLUMN_IDENTIFIER_FILE_ROW_NUMBER, TableColumn("file_row_number", LogicalType::UBIGINT)));
		result.insert(make_pair(_COLUMN_IDENTIFIER_FILE_INDEX, TableColumn("file_index", LogicalType::UBIGINT)));

		scan_bind_data.virtual_col[_COLUMN_IDENTIFIER_FILE_ROW_NUMBER] = "file_row_number";
		scan_bind_data.virtual_col[_COLUMN_IDENTIFIER_FILE_INDEX] = "file_index";

		return result;
	};

	ExtensionUtil::RegisterFunction(instance, vortex_scan);
}

} // namespace vortex
