#![allow(clippy::unwrap_used)]
use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(crate_types) = env::var("CARGO_CFG_CRATE_TYPE") {
        // Skip C header generation if not building a C-compatible library.
        // As of now, we do this in order to not run Miri on the build.rs file.
        if !crate_types.contains("cdylib") && !crate_types.contains("staticlib") {
            println!(
                "cargo:warning=Skipping C header generation (not building C-compatible library)"
            );
            return;
        }
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir).join("cinclude").join("vortex.h");
    println!("{crate_dir}");
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
