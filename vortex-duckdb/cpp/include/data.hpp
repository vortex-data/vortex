// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

// Owns a pointer with a DuckDB delete callback, freeing it on destruction.
class CData final {
public:
    CData(void *data_ptr, duckdb_delete_callback_t callback);
    CData(const CData &) = delete;
    CData &operator=(const CData &) = delete;
    ~CData();
    void *DataPtr() const;

private:
    void *data = nullptr;
    duckdb_delete_callback_t delete_callback = nullptr;
};
