// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "data.hpp"
#include "duckdb/function/copy_function.hpp"

using namespace duckdb;

struct VortexCopyBindData final : TableFunctionData {
    VortexCopyBindData(unique_ptr<CData> ffi_bind, vector<string> column_names)
        : ffi_bind(std::move(ffi_bind)), column_names(std::move(column_names)) {
    }
    unique_ptr<CData> ffi_bind;
    // Column names in write order, used to key WRITTEN_FILE_STATISTICS.
    vector<string> column_names;
};

struct VortexCopyGlobalState final : GlobalFunctionData {
    VortexCopyGlobalState(unique_ptr<CData> ffi_global) : ffi_global(std::move(ffi_global)) {
    }
    unique_ptr<CData> ffi_global;
    // Non-owning; null when the plan does not request statistics.
    CopyFunctionFileStatistics *written_stats = nullptr;
};

struct VortexCopyPreparedBatchData final : PreparedBatchData {
    VortexCopyPreparedBatchData(unique_ptr<CData> ffi_copy_prepared)
        : ffi_copy_prepared(std::move(ffi_copy_prepared)) {
    }
    unique_ptr<CData> ffi_copy_prepared;
};
