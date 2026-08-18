// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "data.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"

using namespace duckdb;

struct VortexBindData final : TableFunctionData {
    VortexBindData() = default;
    unique_ptr<FunctionData> Copy() const override;
    bool Equals(const FunctionData &other) const override;

    unique_ptr<CData> ffi_bind_data;
};

struct VortexBindResult {
    vector<LogicalType> &return_types;
    vector<string> &names;
};

struct VortexGlobalState final : GlobalTableFunctionState {
    VortexGlobalState() = default;
    ~VortexGlobalState() override = default;

    const void *ffi_bind_data = nullptr; // needed for local state partial accumulation
    unique_ptr<CData> ffi_global_state;
};

struct VortexLocalState final : LocalTableFunctionState {
    VortexLocalState() = default;
    unique_ptr<CData> ffi_local_state;
};

struct VortexMultiFileReader final : MultiFileReader {
    inline unique_ptr<MultiFileReader> Copy() const override {
        return make_uniq<VortexMultiFileReader>();
    }

    /*
     * Called after InitializeGlobalState but before TryInitializeScan under
     * file-local lock. Used to avoid initializing file splits if footer
     * statistics prove false for pushed filter or file index is not present in
     * file selection.
     *
     * TODO(myrrc):
     * InitializeReader prunes unused files e.g. ones filtered by "file_index".
     * These files are opened but not initialized. Duckdb 2.0 allows filtering
     * on virtual columns to skip opening files, but until it's released,
     * filtered files (say, second pass in late materialization queries like
     * clickbench 23) will be opened, and we will read their footers.
     *
     * ComplexFilterPushdown/DynamicFilterPushdown callbacks technically allow
     * filtering file list via filter, but with file_index specifically it
     * doesn't work.
     * Say we have a filter "file_index IN (1, 2)" and files 0,1,2. We filter
     * first file and return a new list of two files. Duckdb 1.5 then
     * renumbers files 1->0, 2->1 (it uses file index). Then it evaluates
     * the old filter with new file index and file 1 (with new index 0) is
     * filtered out which produces invalid results.
     */
    ReaderInitializeType InitializeReader(MultiFileReaderData &reader_data,
                                          const MultiFileBindData &bind_data,
                                          const vector<MultiFileColumnDefinition> &global_columns,
                                          const vector<ColumnIndex> &global_column_ids,
                                          optional_ptr<TableFilterSet> table_filters,
                                          ClientContext &context,
                                          MultiFileGlobalState &gstate) override;
};

struct VortexReaderInterface final : MultiFileReaderInterface {
    static unique_ptr<MultiFileReaderInterface> CreateInterface(ClientContext &) {
        return make_uniq<VortexReaderInterface>();
    }

    inline unique_ptr<BaseFileReaderOptions> InitializeOptions(ClientContext &,
                                                               optional_ptr<TableFunctionInfo>) override {
        return make_uniq<BaseFileReaderOptions>();
    }

    inline bool ParseCopyOption(ClientContext &,
                                const string &,
                                const vector<Value> &,
                                BaseFileReaderOptions &,
                                vector<string> &,
                                vector<LogicalType> &) override {
        return false;
    };

    inline bool ParseOption(ClientContext &,
                            const string &,
                            const Value &,
                            MultiFileOptions &,
                            BaseFileReaderOptions &) override {
        return false;
    }

    inline unique_ptr<TableFunctionData> InitializeBindData(MultiFileBindData &,
                                                            unique_ptr<BaseFileReaderOptions>) override {
        return make_uniq<VortexBindData>();
    }

    // Open first file, populate types and names from it
    void BindReader(ClientContext &context,
                    vector<LogicalType> &types,
                    vector<string> &names,
                    MultiFileBindData &bind_data) override;

    unique_ptr<GlobalTableFunctionState> InitializeGlobalState(ClientContext &context,
                                                               MultiFileBindData &bind_data,
                                                               MultiFileGlobalState &global_state) override;

    unique_ptr<LocalTableFunctionState> InitializeLocalState(ExecutionContext &context,
                                                             GlobalTableFunctionState &global_state) override;

    inline shared_ptr<BaseFileReader> CreateReader(ClientContext &,
                                                   GlobalTableFunctionState &,
                                                   BaseUnionData &,
                                                   const MultiFileBindData &) override {
        throw BinderException("UNION BY NAME for Vortex files is not supported");
    }

    shared_ptr<BaseFileReader> CreateReader(ClientContext &context,
                                            GlobalTableFunctionState &gstate,
                                            const OpenFileInfo &file,
                                            idx_t file_idx,
                                            const MultiFileBindData &bind_data) override;

    shared_ptr<BaseFileReader> CreateReader(ClientContext &context,
                                            const OpenFileInfo &file,
                                            BaseFileReaderOptions &options,
                                            const MultiFileOptions &file_options) override;

    unique_ptr<NodeStatistics> GetCardinality(const MultiFileBindData &bind_data, idx_t file_count) override;

    inline FileGlobInput GetGlobInput() override {
        return {FileGlobOptions::FALLBACK_GLOB, "vortex"};
    }

    inline unique_ptr<MultiFileReaderInterface> Copy() override {
        return make_uniq<VortexReaderInterface>();
    }

    void GetVirtualColumns(ClientContext &, MultiFileBindData &, virtual_column_map_t &result) override;
    bool FinalizeScan(ClientContext &, GlobalTableFunctionState &gstate, DataChunk &output) override;
    void FinishReading(ClientContext &,
                       GlobalTableFunctionState &gstate,
                       LocalTableFunctionState &lstate) override;
};

struct VortexBaseReader final : BaseFileReader {
    VortexBaseReader(OpenFileInfo file, unique_ptr<CData> ffi_file)
        : BaseFileReader(file), ffi_file(std::move(ffi_file)) {
    }

    unique_ptr<CData> ffi_file;
    vector<column_t> virtual_ids;

    inline void AddVirtualColumn(column_t id) override {
        virtual_ids.push_back(id);
    }

    /*
     * Called by all threads on current file under global lock. Once
     * TryInitializeScan returns false, first thread to receive it advances
     * to next file and calls TryInitializeScan on it.
     */
    bool TryInitializeScan(ClientContext &,
                           GlobalTableFunctionState &,
                           LocalTableFunctionState &local_state) override;

    AsyncResult Scan(ClientContext &,
                     GlobalTableFunctionState &global_state,
                     LocalTableFunctionState &local_state,
                     DataChunk &chunk) override;

    inline void FinishFile(ClientContext &, GlobalTableFunctionState &) override {
    }

    double GetProgressInFile(ClientContext &) override;

    unique_ptr<BaseStatistics> GetStatistics(ClientContext &context, const string &name) override;

    inline string GetReaderType() const override {
        return "Vortex";
    }
};
