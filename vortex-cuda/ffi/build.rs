// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::error::Error;
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::process;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let header = "cinclude/vortex_cuda.h";
    let mut generated = Vec::new();
    // Parse only the FFI source to avoid macro expansion and recursive CUDA builds.
    cbindgen::Builder::new()
        .with_src("src/lib.rs")
        .with_config(cbindgen::Config::from_file("cbindgen.toml")?)
        .generate()?
        .write(&mut generated);
    match fs::read(header) {
        Ok(existing) if existing == generated => return Ok(()),
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failed to read {header}: {error}").into()),
    }
    publish_header(header, &generated)
        .map_err(|error| format!("failed to publish {header}: {error}"))?;
    Ok(())
}

fn publish_header(header: &str, generated: &[u8]) -> io::Result<()> {
    let mut attempt = 0_u64;
    let (temporary, mut file) = loop {
        let temporary = format!("{header}.{}.{attempt}.tmp", process::id());
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => attempt += 1,
            Err(error) => return Err(error),
        }
    };

    // Publish atomically so concurrent builds never see a truncated header.
    let written = file.write_all(generated);
    drop(file);
    let result = written.and_then(|()| fs::rename(&temporary, header));
    if result.is_err()
        && let Err(error) = fs::remove_file(&temporary)
    {
        eprintln!("failed to remove temporary header {temporary}: {error}");
    }
    result
}
