// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

// Main header file for vortex-clickhouse C++ components.
//
// This header provides the C API for integrating Vortex with ClickHouse.
// Include this header in your ClickHouse plugin code.
//
// The API is organized into several components:
//
// - common.h: Basic types and handles
// - scanner.h: Reading Vortex files
// - exporter.h: Extracting data from Vortex arrays
// - writer.h: Writing Vortex files
// - column.h: Column type utilities
// - format.h: Format constants
//
// Example usage (reading):
//
//   #include <clickhouse_vx.h>
//
//   // Open a Vortex file
//   VortexScanner* scanner = vortex_scanner_new("/path/to/data.vortex");
//   if (!scanner) { return handleError(); }
//
//   // Get schema
//   size_t num_cols = vortex_scanner_num_columns(scanner);
//   for (size_t i = 0; i < num_cols; i++) {
//       const char* name = vortex_scanner_column_name(scanner, i);
//       const char* type = vortex_scanner_column_type(scanner, i);
//       // ... configure ClickHouse columns ...
//   }
//
//   // Read data
//   while (vortex_scanner_has_more(scanner)) {
//       VortexExporterHandle* batch = vortex_scanner_read_batch(scanner);
//       while (vortex_exporter_has_more(batch)) {
//           // Export to ClickHouse columns
//           vortex_exporter_export(batch, buffer, max_rows);
//       }
//       vortex_exporter_free(batch);
//   }
//
//   vortex_scanner_free(scanner);

#include "clickhouse_vx/common.h"
#include "clickhouse_vx/format.h"
#include "clickhouse_vx/column.h"
#include "clickhouse_vx/scanner.h"
#include "clickhouse_vx/exporter.h"
#include "clickhouse_vx/writer.h"
