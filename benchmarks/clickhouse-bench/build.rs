// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Build script that exports the ClickHouse binary path.
//!
//! Resolution order:
//! 1. `CLICKHOUSE_BINARY` env var — use as-is.
//! 2. Falls back to `"clickhouse"` (i.e., resolve from `$PATH` at runtime).
//!
//! Users must install ClickHouse themselves for local runs.
//! In CI, it is installed via the workflow before the benchmark step.

fn main() {
    println!("cargo:rerun-if-env-changed=CLICKHOUSE_BINARY");

    let binary = std::env::var("CLICKHOUSE_BINARY").unwrap_or_else(|_| "clickhouse".to_string());
    println!("cargo:rustc-env=CLICKHOUSE_BINARY={binary}");
}
