// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/vector.hpp"

#include "duckdb_vx.h"
#include "duckdb_vx/data.hpp"
#include "duckdb_vx/vector_buffer.hpp"

using namespace duckdb;

extern "C" duckdb_vx_vector_buffer duckdb_vx_vector_buffer_create(duckdb_vx_data buffer) {
    auto data = reinterpret_cast<vortex::CData *>(buffer);
    auto *shared_buffer = new duckdb::shared_ptr<vortex::ExternalVectorBuffer>(
        duckdb::make_shared_ptr<vortex::ExternalVectorBuffer>(unique_ptr<vortex::CData>(data)));
    return reinterpret_cast<duckdb_vx_vector_buffer>(shared_buffer);
}

extern "C" void duckdb_vx_vector_buffer_destroy(duckdb_vx_vector_buffer *buffer) {
    if (buffer != nullptr && *buffer != nullptr) {
        auto shared_buffer = reinterpret_cast<shared_ptr<vortex::ExternalVectorBuffer> *>(*buffer);
        delete shared_buffer;
        *buffer = nullptr;
    }
}
