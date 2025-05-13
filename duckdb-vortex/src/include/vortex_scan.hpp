#pragma once

#include "duckdb/main/extension_util.hpp"

namespace vortex {
void RegisterScanFunction(duckdb::DatabaseInstance &instance);
}
