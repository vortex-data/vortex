// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::path::PathBuf;

use get_dir::FileTarget;
use get_dir::GetDir;
use get_dir::Target;

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
    let target_dir = workspace_root().join("target");
    let lib_path = target_dir.join("duckdb-lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path.display());
}

fn workspace_root() -> PathBuf {
    GetDir::new()
        .target(Target::File(FileTarget::new("Cargo.lock")))
        .run_reverse()
        .expect("Can't find workspace root")
}
