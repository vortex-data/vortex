// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "data.hpp"
#include "duckdb/common/vector/flat_vector.hpp"

using namespace duckdb;

// A DuckDB vector buffer over externally-owned data.
class ExternalVectorBuffer final : public StandardVectorBuffer {
    shared_ptr<ExternalVectorBuffer> parent;
    unique_ptr<CData> data;

public:
    explicit inline ExternalVectorBuffer(unique_ptr<CData> data)
        : StandardVectorBuffer(nullptr, count_t(0), 0), data(std::move(data)) {
    }

    inline ExternalVectorBuffer(shared_ptr<ExternalVectorBuffer> parent,
                                data_ptr_t ptr,
                                idx_t capacity,
                                idx_t type_size)
        : StandardVectorBuffer(ptr, count_t(capacity), type_size), parent(std::move(parent)) {
    }
};
