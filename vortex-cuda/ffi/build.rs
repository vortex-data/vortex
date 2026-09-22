// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::error::Error;
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::process;
use std::process::Command;
use std::process::Stdio;
use std::thread;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../.clang-format");

    let header = "cinclude/vortex_cuda.h";
    let mut generated = Vec::new();
    // The CUDA API needs no macro expansion or dependency parsing, so generate on stable Rust
    // without recursively building the CUDA implementation.
    cbindgen::Builder::new()
        .with_src("src/lib.rs")
        .with_config(cbindgen::Config::from_file("cbindgen.toml")?)
        .generate()?
        .write(&mut generated);
    let formatted = format_header(header, generated)?;
    match fs::read(header) {
        Ok(existing) if existing == formatted => return Ok(()),
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failed to read {header}: {error}").into()),
    }
    publish_header(header, &formatted)
        .map_err(|error| format!("failed to publish {header}: {error}"))?;
    Ok(())
}

fn format_header(header: &str, generated: Vec<u8>) -> Result<Vec<u8>, Box<dyn Error>> {
    // Cargo runs this script in the crate directory; the assumed header path lets clang-format
    // find the repository's .clang-format while reading from stdin.
    let mut child = Command::new("clang-format")
        .arg("--style=file")
        .arg(format!("--assume-filename={header}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!("failed to start clang-format (required on PATH to generate {header}): {error}")
        })?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or("clang-format stdin is unavailable")?;
    // Feed stdin while wait_with_output drains stdout and stderr, even when a header exceeds
    // pipe capacity. Dropping stdin in the writer signals EOF to clang-format.
    let writer = thread::spawn(move || stdin.write_all(&generated));
    let output = child.wait_with_output();
    let written = writer
        .join()
        .map_err(|_| "clang-format stdin writer panicked")?;
    let output =
        output.map_err(|error| format!("failed to collect clang-format output: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "clang-format failed for {header} ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    written.map_err(|error| format!("failed to write header to clang-format stdin: {error}"))?;
    Ok(output.stdout)
}

fn publish_header(header: &str, formatted: &[u8]) -> io::Result<()> {
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

    // A unique sibling file and atomic rename prevent concurrent builds from exposing a
    // truncated shared header. Close the file before renaming or cleaning it up.
    let written = file.write_all(formatted);
    drop(file);
    let result = written.and_then(|()| fs::rename(&temporary, header));
    if result.is_err()
        && let Err(error) = fs::remove_file(&temporary)
    {
        eprintln!("failed to remove temporary header {temporary}: {error}");
    }
    result
}
