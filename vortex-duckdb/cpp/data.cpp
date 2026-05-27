// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb.h"

#include "duckdb_vx/data.hpp"

namespace vortex {

CData::CData(void *data_ptr, duckdb_delete_callback_t callback) : data(data_ptr), delete_callback(callback) {
}

CData::~CData() {
    if (data && delete_callback) {
        delete_callback(data);
    }
    data = nullptr;
    delete_callback = nullptr;
}

void *CData::DataPtr() const {
    return data;
}

extern "C" duckdb_vx_data duckdb_vx_data_create(void *data, duckdb_delete_callback_t delete_callback) {
    return reinterpret_cast<duckdb_vx_data>(new CData(data, delete_callback));
}

} // namespace vortex
