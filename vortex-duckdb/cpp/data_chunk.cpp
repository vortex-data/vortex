// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/data_chunk.h"
#include "duckdb/common/types/data_chunk.hpp"

const char *duckdb_data_chunk_to_string(duckdb_data_chunk chunk, duckdb_vx_error *err) {
    try {
        auto dchunk = reinterpret_cast<duckdb::DataChunk *>(chunk);
        auto str = dchunk->ToString();
        auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
        memcpy(result, str.c_str(), str.size() + 1);
        *err = nullptr;
        return result;
    } catch (std::runtime_error &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
        return nullptr;
    }
}

void duckdb_data_chunk_verify(duckdb_data_chunk chunk, duckdb_vx_error *err) {
    try {
        auto dchunk = reinterpret_cast<duckdb::DataChunk *>(chunk);
        dchunk->Verify();
        *err = nullptr;
    } catch (std::runtime_error &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
    }
}
