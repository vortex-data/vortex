// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

fn main() {
    // Propagate DuckDB rpath from vortex-duckdb
    let duckdb_lib = std::env::var("DEP_DUCKDB_LIB_DIR").unwrap();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{duckdb_lib}");
}
