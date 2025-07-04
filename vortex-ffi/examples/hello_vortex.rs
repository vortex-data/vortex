// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used, clippy::expect_used)]
//! This example shows usage of the Vortex C FFI to read a Vortex file written by a Rust client.
//!
//! You can invoke this example from a checkout by running
//!
//! ```ignore
//!cargo run -p vortex-ffi --example hello_vortex
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use tempfile::NamedTempFile;
use tokio::fs::File as TokioFile;
use tokio::runtime::Runtime;
use vortex::arrays::{ChunkedArray, StructArray};
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::file::VortexWriteOptions;
use vortex::{Array, ArrayRef, IntoArray};

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| Runtime::new().unwrap());

const BIN_NAME: &str = "hello_vortex";

pub fn main() -> VortexResult<()> {
    let bin_path = PathBuf::new()
        .join(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(BIN_NAME);

    let exit = Command::new("cc")
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
        .status()?;

    assert!(exit.success());

    let file = NamedTempFile::with_suffix(".vortex")?.into_temp_path();

    // Write the test data
    RUNTIME.block_on(write_vortex_file(&file))?;

    let uri = format!("file://{}", file.display());

    // Invoke the binary we just created
    let exit = Command::new(&bin_path).arg(uri).status()?;
    assert!(exit.success());

    Ok(())
}

async fn write_vortex_file(path: impl AsRef<Path>) -> VortexResult<()> {
    let file = TokioFile::create(path).await?;

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

    VortexWriteOptions::default()
        .write(file, test_data.to_array_stream())
        .await?;

    Ok(())
}

fn chunk(nums: Buffer<i32>, floats: Buffer<f32>) -> ArrayRef {
    StructArray::try_from_iter([("nums", nums.into_array()), ("floats", floats.into_array())])
        .unwrap()
        .into_array()
}
