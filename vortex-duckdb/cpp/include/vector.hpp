// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "data.hpp"
#include "duckdb/common/types/vector_buffer.hpp"

// A DuckDB vector buffer that keeps externally-owned data alive for the
// lifetime of the vector.
class ExternalVectorBuffer final : public duckdb::VectorBuffer {
    duckdb::unique_ptr<CData> data;

public:
    explicit inline ExternalVectorBuffer(duckdb::unique_ptr<CData> data) : data(std::move(data)) {
    }
};
