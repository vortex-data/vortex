// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
use std::env;
use std::path::PathBuf;

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
    const DUCKDB_VERSION: &str = "v1.3.2";
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_dir = manifest_dir.parent().unwrap().join("target");
    let lib_path = target_dir.join(format!("duckdb-lib-{DUCKDB_VERSION}"));
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path.display());
}
