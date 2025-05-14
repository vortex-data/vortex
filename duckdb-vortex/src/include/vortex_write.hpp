#pragma once

#include "duckdb/main/extension_util.hpp"

namespace vortex {
void RegisterWriteFunction(duckdb::DatabaseInstance &instance);
}
