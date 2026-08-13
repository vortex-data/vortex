// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "multi_file_reader.hpp"
#include "error.hpp"
#include "table_function.h"
#include "vortex_duckdb.h"
#include "vortex.h"

// TODO (myrrc) remove NDEBUG in release builds

unique_ptr<FunctionData> VortexBindData::Copy() const {
    auto result = make_uniq<VortexBindData>();
    if (ffi_bind_data) {
        const duckdb_vx_data copy = duckdb_table_function_bind_data_clone(ffi_bind_data->DataPtr());
        result->ffi_bind_data = unique_ptr<CData>(reinterpret_cast<CData *>(copy));
    }
    return result;
}

bool VortexBindData::Equals(const FunctionData &other_base) const {
    const VortexBindData &other = other_base.Cast<VortexBindData>();
    return ffi_bind_data.get() == other.ffi_bind_data.get();
}

ReaderInitializeType
VortexMultiFileReader::InitializeReader(MultiFileReaderData &reader_data,
                                        const MultiFileBindData &,
                                        const vector<MultiFileColumnDefinition> &global_columns,
                                        const vector<ColumnIndex> &,
                                        optional_ptr<TableFilterSet>,
                                        ClientContext &,
                                        MultiFileGlobalState &gstate) {
    D_ASSERT(gstate.global_state != nullptr);

    // TODO(myrrc): call MultiFileReader::InitializeReader and exit early once
    // we support hive partitioning and/or UNION BY NAME

    // projection expression pushdown and aggregate pushdown may have
    // changed columns, update them in our reader
    VortexBaseReader &reader = reader_data.reader->Cast<VortexBaseReader>();
    reader.columns = global_columns;

    VortexGlobalState &global = gstate.global_state->Cast<VortexGlobalState>();
    duckdb_vx_error error = nullptr;
    void *const ffi_global_state = global.ffi_global_state->DataPtr();
    void *const ffi_file = reader.ffi_file->DataPtr();
    const bool skip = duckdb_table_function_file_should_skip(ffi_global_state, ffi_file, &error);
    if (error) {
        throw InvalidInputException(IntoErrString(error));
    }
    return skip ? ReaderInitializeType::SKIP_READING_FILE : ReaderInitializeType::INITIALIZED;
}

void VortexReaderInterface::BindReader(ClientContext &,
                                       vector<LogicalType> &,
                                       vector<string> &,
                                       MultiFileBindData &bind_data) {
    auto &bind = bind_data.bind_data->Cast<VortexBindData>();
    VortexBaseReader &initial_reader = bind_data.initial_reader->Cast<VortexBaseReader>();
    duckdb_vx_error error = nullptr;
    void *const ffi_file = initial_reader.ffi_file->DataPtr();
    duckdb_vx_data ffi_bind_data = duckdb_table_function_bind(ffi_file, &error);
    if (error) {
        throw BinderException(IntoErrString(error));
    }
    bind.ffi_bind_data = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data));
}

unique_ptr<GlobalTableFunctionState>
VortexReaderInterface::InitializeGlobalState(ClientContext &context,
                                             MultiFileBindData &bind_data,
                                             MultiFileGlobalState &input) {
    void *const ffi_bind = bind_data.bind_data->Cast<VortexBindData>().ffi_bind_data->DataPtr();

    // Pushed projection expressions and casts change what the scan outputs
    vector<LogicalType> types;
    vector<string> names;
    VortexBindResult schema = {types, names};
    duckdb_table_function_bind_schema(ffi_bind, reinterpret_cast<duckdb_vx_tfunc_bind_result>(&schema));
    bind_data.columns = MultiFileColumnDefinition::ColumnsFromNamesAndTypes(names, types);

    vector<idx_t> column_ids(input.column_indexes.size());
    for (size_t i = 0; i < input.column_indexes.size(); ++i) {
        column_ids[i] = input.column_indexes[i].GetPrimaryIndex();
    }

    duckdb_vx_tfunc_init_input ffi_input = {
        .bind_data = ffi_bind,
        .column_ids = column_ids.data(),
        .column_ids_count = column_ids.size(),
        .filters = reinterpret_cast<duckdb_vx_table_filter_set>(input.filters.get()),
        .client_context = reinterpret_cast<duckdb_client_context>(&context),
    };

    duckdb_vx_error error_out = nullptr;
    duckdb_vx_data ffi_global_state = duckdb_table_function_init_global(&ffi_input, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto result = make_uniq<VortexGlobalState>();
    result->ffi_bind_data = ffi_bind;
    result->ffi_global_state = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global_state));
    return result;
}

unique_ptr<LocalTableFunctionState>
VortexReaderInterface::InitializeLocalState(ExecutionContext &, GlobalTableFunctionState &global_state) {
    auto &global = global_state.Cast<VortexGlobalState>();
    duckdb_vx_data ffi_local_state =
        duckdb_table_function_init_local(global.ffi_bind_data, global.ffi_global_state->DataPtr());

    auto result = make_uniq<VortexLocalState>();
    result->ffi_local_state = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_local_state));
    return result;
}

static shared_ptr<BaseFileReader> OpenReader(const OpenFileInfo &file, idx_t file_idx) {
    duckdb_vx_error error = nullptr;
    duckdb_vx_data ffi_file =
        duckdb_table_function_file_open(file.path.c_str(), file.path.size(), file_idx, &error);
    if (error) {
        throw IOException(IntoErrString(error));
    }
    return make_shared_ptr<VortexBaseReader>(file, unique_ptr<CData>(reinterpret_cast<CData *>(ffi_file)));
}

// open file, read schema
shared_ptr<BaseFileReader> VortexReaderInterface::CreateReader(ClientContext &,
                                                               GlobalTableFunctionState &,
                                                               const OpenFileInfo &file,
                                                               idx_t file_idx,
                                                               const MultiFileBindData &) {
    return OpenReader(file, file_idx);
}

shared_ptr<BaseFileReader> VortexReaderInterface::CreateReader(ClientContext &,
                                                               const OpenFileInfo &file,
                                                               BaseFileReaderOptions &,
                                                               const MultiFileOptions &) {
    auto reader = OpenReader(file, 0);

    vector<LogicalType> types;
    vector<string> names;
    VortexBindResult schema = {types, names};
    duckdb_vx_error error_out = nullptr;
    duckdb_table_function_file_schema(reader->Cast<VortexBaseReader>().ffi_file->DataPtr(),
                                      reinterpret_cast<duckdb_vx_tfunc_bind_result>(&schema),
                                      &error_out);
    if (error_out) {
        throw IOException(IntoErrString(error_out));
    }
    reader->Cast<VortexBaseReader>().columns =
        MultiFileColumnDefinition::ColumnsFromNamesAndTypes(names, types);
    return reader;
}

unique_ptr<NodeStatistics> VortexReaderInterface::GetCardinality(const MultiFileBindData &data,
                                                                 idx_t file_count) {
    const VortexBindData &bind_data = data.bind_data->Cast<VortexBindData>();
    const void *const ffi_bind = bind_data.ffi_bind_data->DataPtr();

    duckdb_vx_node_statistics stats = {};
    duckdb_table_function_cardinality(ffi_bind, file_count, &stats);

    auto out = make_uniq<NodeStatistics>();
    out->has_estimated_cardinality = stats.has_estimated_cardinality;
    out->estimated_cardinality = stats.estimated_cardinality;
    out->has_max_cardinality = stats.has_max_cardinality;
    out->max_cardinality = stats.max_cardinality;
    return out;
}

void VortexBaseReader::PrepareReader(ClientContext &, GlobalTableFunctionState &state) {
    /*
     * Start the scan early. Default MultiFileReader model is "1 split per
     * worker". This works poorly for queries on multiple files, think
     * clickbench. What we do here is we launch prefetch/decode for the file
     * before reader has been bound. This means by the time a worker starts
     * processing next file, next file' splits are already decoded.
     *
     * Another performance consideration is that TryInitializeScan is called
     * under global lock and PrepareReader is called under file-local lock.
     *
     * With duckdb 2.0 this code will migrate to readahead queue.
     */
    VortexGlobalState &global = state.Cast<VortexGlobalState>();

    const idx_t real_columns = columns.size() - virtual_ids.size();
    vector<idx_t> local_column_ids(column_ids.size());
    for (idx_t i = 0; i < column_ids.size(); i++) {
        const idx_t local_id = column_ids[MultiFileLocalIndex(i)];
        local_column_ids[i] = local_id < real_columns ? local_id : virtual_ids[local_id - real_columns];
    }

    duckdb_vx_error error_out = nullptr;
    duckdb_vx_data ffi_scan =
        duckdb_table_function_file_start_scan(global.ffi_bind_data,
                                              global.ffi_global_state->DataPtr(),
                                              ffi_file->DataPtr(),
                                              local_column_ids.data(),
                                              local_column_ids.size(),
                                              reinterpret_cast<duckdb_vx_table_filter_set>(filters.get()),
                                              &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
    ffi_file_scan = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_scan));
}

bool VortexBaseReader::TryInitializeScan(ClientContext &,
                                         GlobalTableFunctionState &,
                                         LocalTableFunctionState &) {
    // false if file is exhausted
    D_ASSERT(ffi_file_scan != nullptr); // scan already started in PrepareReader
    return duckdb_table_function_file_has_work(ffi_file_scan->DataPtr());
}

AsyncResult VortexBaseReader::Scan(ClientContext &,
                                   GlobalTableFunctionState &global_state,
                                   LocalTableFunctionState &local_state,
                                   DataChunk &chunk) {
    auto &local = local_state.Cast<VortexLocalState>();

    duckdb_data_chunk ffi_chunk = reinterpret_cast<duckdb_data_chunk>(&chunk);
    duckdb_vx_error error_out = nullptr;
    duckdb_table_function_file_scan(ffi_file_scan->DataPtr(),
                                    global_state.Cast<VortexGlobalState>().ffi_global_state->DataPtr(),
                                    local.ffi_local_state->DataPtr(),
                                    ffi_chunk,
                                    &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }

    if (chunk.size() == 0) {
        return SourceResultType::FINISHED;
    }
    return SourceResultType::HAVE_MORE_OUTPUT;
}

void VortexReaderInterface::GetVirtualColumns(ClientContext &,
                                              MultiFileBindData &,
                                              virtual_column_map_t &result) {
    // "filename", "file_index" and "empty" come from MultiFileReader
    result.insert(
        {MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER, {"file_row_number", LogicalType::UBIGINT}});
}

bool VortexReaderInterface::FinalizeScan(ClientContext &,
                                         GlobalTableFunctionState &gstate,
                                         DataChunk &output) {
    auto &global = gstate.Cast<VortexGlobalState>();
    duckdb_data_chunk ffi_chunk = reinterpret_cast<duckdb_data_chunk>(&output);
    duckdb_vx_error error_out = nullptr;
    const bool filled =
        duckdb_table_function_finalize_scan(global.ffi_global_state->DataPtr(), ffi_chunk, &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
    return filled;
}

static Value &UnwrapValue(duckdb_value value) {
    return *(reinterpret_cast<Value *>(value));
}

static unique_ptr<BaseStatistics> numeric_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = NumericStats::CreateUnknown(type);
    if (stats.min) {
        NumericStats::SetMin(out, UnwrapValue(stats.min));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        NumericStats::SetMax(out, UnwrapValue(stats.max));
        duckdb_destroy_value(&stats.max);
    }
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

static unique_ptr<BaseStatistics> string_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (stats.min) {
        StringStats::SetMin(out, StringValue::Get(UnwrapValue(stats.min)));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        StringStats::SetMax(out, StringValue::Get(UnwrapValue(stats.max)));
        duckdb_destroy_value(&stats.max);
    }
    if (stats.max_string_length >> 63) {
        StringStats::SetMaxStringLength(out, uint32_t(stats.max_string_length));
    }
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }

    return out.ToUnique();
}

static unique_ptr<BaseStatistics> base_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = BaseStatistics::CreateUnknown(type);
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

unique_ptr<BaseStatistics> VortexBaseReader::GetStatistics(ClientContext &, const string &name) {
    duckdb_column_statistics statistics = {};
    if (!duckdb_table_function_file_statistics(ffi_file->DataPtr(), name.c_str(), name.size(), &statistics)) {
        return nullptr;
    }
    for (const auto &column : columns) {
        if (column.name != name) {
            continue;
        }
        const LogicalType &type = column.type;
        switch (type.id()) {
        case LogicalTypeId::BOOLEAN:
        case LogicalTypeId::TINYINT:
        case LogicalTypeId::SMALLINT:
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT:
        case LogicalTypeId::FLOAT:
        case LogicalTypeId::DOUBLE:
        case LogicalTypeId::UTINYINT:
        case LogicalTypeId::USMALLINT:
        case LogicalTypeId::UINTEGER:
        case LogicalTypeId::UBIGINT:
        case LogicalTypeId::UHUGEINT:
        case LogicalTypeId::HUGEINT: {
            return numeric_stats(statistics, type);
        }
        case LogicalTypeId::VARCHAR:
        case LogicalTypeId::BLOB: {
            return string_stats(statistics, type);
        }
        case LogicalTypeId::STRUCT: {
            // TODO(myrrc)
            // Duckdb's has_null has a different semantics for structs.
            // If we propagate our has_null, this breaks Duckdb optimizer.
            // You can reproduce it in struct.slt test in vortex-sqllogictests:
            return {};
        }
        default:
            return base_stats(statistics, type);
        }
    }
    return nullptr;
}

double VortexBaseReader::GetProgressInFile(ClientContext &) {
    if (!ffi_file_scan) {
        return 0.0;
    }
    return duckdb_table_function_file_progress(ffi_file_scan->DataPtr());
}
