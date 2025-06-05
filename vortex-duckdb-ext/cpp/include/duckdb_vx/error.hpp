#pragma once

#include <string>
#include "duckdb_vx/error.h"

namespace vortex {
std::string IntoErrString(duckdb_vx_error error);
}
