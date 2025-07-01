#![allow(clippy::unwrap_used)]
use std::env;
use std::path::PathBuf;

#[cfg(not(miri))]
fn main() {
    // Skip cbindgen under Miri.
    if env::var("MIRI").is_ok() {
        println!("cargo:warning=Skipping C header generation under miri");
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir).join("cinclude").join("vortex.h");
    std::fs::create_dir_all(output_file.parent().unwrap()).unwrap();

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_file("cbindgen.toml").unwrap())
        .generate()
        .unwrap()
        .write_to_file(&output_file);

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
