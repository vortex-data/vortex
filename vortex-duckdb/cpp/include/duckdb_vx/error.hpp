// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <string>
#include "duckdb_vx/error.h"

namespace vortex {
std::string IntoErrString(duckdb_vx_error error);
}
