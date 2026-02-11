// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "common.h"

// Format-related constants for ClickHouse integration.

#ifdef __cplusplus
extern "C" {
#endif

/// Format name constant used in ClickHouse registration.
extern const char* VORTEX_FORMAT_NAME;

/// Default file extension for Vortex files.
extern const char* VORTEX_FILE_EXTENSION;

/// Magic bytes at the start of a Vortex file.
extern const uint8_t VORTEX_MAGIC_BYTES[4];

/// Current Vortex format version.
extern const uint32_t VORTEX_FORMAT_VERSION;

#ifdef __cplusplus
}
#endif
