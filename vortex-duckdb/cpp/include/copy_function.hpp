// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "data.hpp"
#include "duckdb/function/copy_function.hpp"

using namespace duckdb;

struct VortexCopyBindData final : TableFunctionData {
    VortexCopyBindData(unique_ptr<CData> ffi_bind) : ffi_bind(std::move(ffi_bind)) {
    }
    unique_ptr<CData> ffi_bind;
};

struct VortexCopyGlobalState final : GlobalFunctionData {
    VortexCopyGlobalState(unique_ptr<CData> ffi_global) : ffi_global(std::move(ffi_global)) {
    }
    unique_ptr<CData> ffi_global;
};

struct VortexCopyPreparedBatchData final : PreparedBatchData {
    VortexCopyPreparedBatchData(unique_ptr<CData> ffi_copy_prepared)
        : ffi_copy_prepared(std::move(ffi_copy_prepared)) {
    }
    unique_ptr<CData> ffi_copy_prepared;
};
