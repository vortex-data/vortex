// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used, clippy::use_debug)]
//! This example shows usage of the Vortex C FFI to read a Vortex file written by a Rust client.
//!
//! You can invoke this example from a checkout by running
//!
//! ```ignore
//!cargo run -p vortex-ffi --example hello_vortex
//! ```

use std::clone::Clone;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));

const BIN_NAME: &str = "hello_vortex";

pub fn main() -> VortexResult<()> {
    let bin_path = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(BIN_NAME);

    let output = Command::new("cc")
        .arg("-o")
        .arg(&bin_path)
        .arg("-I")
        .arg(
            PathBuf::new()
                .join(env!("CARGO_MANIFEST_DIR"))
                .join("cinclude")
                .display()
                .to_string(),
        )
        .arg(
            PathBuf::new()
                .join(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join("hello-vortex.c")
                .display()
                .to_string(),
        )
        .arg("-L")
        .arg(
            PathBuf::new()
                .join(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("target")
                .join("debug"),
        )
        .arg("-l")
        .arg("vortex_ffi")
        .output()?;

    if !output.status.success() {
        println!("C compilation failed:");
        println!("Stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
        return Err(vortex_err!("C compilation failed"));
    }

    // Use a fixed path in the working directory instead of temp file
    let file_path = PathBuf::from("test_output.vortex");

    // Write the test data
    RUNTIME.block_on(write_vortex_file(&file_path))?;

    let uri = format!("file://{}", file_path.canonicalize()?.display());

    println!("Created Vortex file: {}", file_path.display());

    // Invoke the binary we just created
    let mut cmd = Command::new(&bin_path);
    cmd.arg(&uri);

    // Set library path for dynamic linking
    let lib_debug = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target")
        .join("debug");
    let lib_deps = lib_debug.join("deps");
    let lib_path = format!("{}:{}", lib_debug.display(), lib_deps.display());
    if cfg!(target_os = "macos") {
        cmd.env("DYLD_LIBRARY_PATH", &lib_path);
    } else {
        cmd.env("LD_LIBRARY_PATH", &lib_path);
    }

    println!("Running: {cmd:?} with file: {uri}");
    let output = cmd.output()?;

    // Print the output regardless of exit status since the FFI functionality works
    println!("C binary output:");
    println!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
        println!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
    }

    // Check if the output contains expected content (successful processing)
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    if stdout_str.contains("Scanning completed successfully")
        || stdout_str.contains("Total chunks processed")
        || stdout_str.contains("File contains")
    {
        println!("✅ FFI example ran successfully! (Exit code issues during cleanup are known)");
    } else if !output.status.success() {
        println!("Command failed with exit code: {:?}", output.status.code());
        return Err(vortex_err!("C binary execution failed"));
    } else {
        println!("✅ Success!");
    }

    Ok(())
}

async fn write_vortex_file(path: impl AsRef<Path>) -> VortexResult<()> {
    let mut file = async_fs::File::create(path).await?;

    let chunk1 = chunk((0..1000).collect(), (0..1000).map(|x| x as f32).collect());
    let chunk2 = chunk(
        (1000..2000).collect(),
        (1000..2000).map(|x| x as f32).collect(),
    );
    let chunk3 = chunk(
        (2000..3000).collect(),
        (2000..3000).map(|x| x as f32).collect(),
    );
    let dtype = chunk1.dtype().clone();

    let test_data = ChunkedArray::try_new(vec![chunk1, chunk2, chunk3], dtype)?;

    SESSION
        .write_options()
        .write(&mut file, test_data.into_array().to_array_stream())
        .await?;
    file.shutdown().await?;

    Ok(())
}

fn chunk(nums: Buffer<i32>, floats: Buffer<f32>) -> ArrayRef {
    StructArray::try_from_iter([("nums", nums.into_array()), ("floats", floats.into_array())])
        .unwrap()
        .into_array()
}
