// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "data.hpp"
#include "vortex_duckdb.h"
#include "table_function.h"
#include "vortex.h"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/parallel/task_executor.hpp"

using namespace duckdb;

struct VortexCopyBindData final : TableFunctionData {
    VortexCopyBindData(unique_ptr<CData> ffi_bind) : ffi_bind(std::move(ffi_bind)) {
    }
    unique_ptr<CData> ffi_bind;
};

struct VortexCopyGlobalState final : GlobalFunctionData {
    VortexCopyGlobalState(unique_ptr<CData> ffi_global, unique_ptr<TaskExecutor> executor)
        : ffi_global(std::move(ffi_global)), executor(std::move(executor)) {
    }

    ~VortexCopyGlobalState() {
        duckdb_copy_function_request_tasks(ffi_global->DataPtr(), 0);
    }

    unique_ptr<CData> ffi_global;
    unique_ptr<TaskExecutor> executor;
    CopyFunctionFileStatistics *written_stats = nullptr;
};

struct VortexCopyPreparedBatchData final : PreparedBatchData {
    VortexCopyPreparedBatchData(unique_ptr<CData> ffi_copy_prepared)
        : ffi_copy_prepared(std::move(ffi_copy_prepared)) {
    }
    unique_ptr<CData> ffi_copy_prepared;
};
