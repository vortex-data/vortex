// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

/// Adds a dynamic linker runtime path pointing to the DuckDB dylib dir.
///
/// Setting an absolute rpath, if required by multiple binaries, is the most
/// robust solution compared to using relative paths in terms of
/// `-rpath,$ORIGIN` or `-rpath,@executable_path`.
///
/// Using an absolute rpath implies that binaries linking against the dynamic
/// DuckDB library are never published.
///
/// Note that the rpath set in vortex-duckdb's build.rs is not inherited by
/// crates linking against it which is why consumers must set a rpath on their end.
///
/// The dynamic DuckDB library is preferred over the static version, as DuckDB's
/// static lib is not self-contained. This means that it includes symbols which
/// are not defined as part of the static library.
fn main() {
    // Propagate DuckDB rpath from vortex-duckdb
    let duckdb_lib = std::env::var("DEP_DUCKDB_LIB_DIR").unwrap();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{duckdb_lib}");
}
