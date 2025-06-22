#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
use std::env;
use std::path::PathBuf;

fn main() {
    const DUCKDB_VERSION: &str = "v1.3.0";

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_dir = manifest_dir.parent().unwrap().join("target");
    let lib_path = target_dir.join(format!("duckdb-{DUCKDB_VERSION}"));
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path.display());
}
