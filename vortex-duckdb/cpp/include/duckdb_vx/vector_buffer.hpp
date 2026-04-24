// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/vector.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx/data.hpp"

namespace vortex {

// This is a wrapper around an externally managed buffer, which can be assigned to a Vector and
// freed once the vector is done with the buffer.
class ExternalVectorBuffer : public duckdb::VectorBuffer {
public:
    explicit ExternalVectorBuffer(duckdb::unique_ptr<CData> data) : data(std::move(data)) {
    }

private:
    duckdb::unique_ptr<CData> data;
};

} // namespace vortex