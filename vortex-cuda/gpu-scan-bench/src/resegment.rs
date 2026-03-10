// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

//! Reads a CUDA-compatible Vortex file and rewrites it with a different segment size.

use std::path::PathBuf;

use clap::Parser;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::VortexSessionDefault;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_cuda::layout::register_cuda_layout;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[derive(Parser)]
#[command(
    name = "resegment",
    about = "Rewrite a CUDA-compatible Vortex file with a different segment size"
)]
struct Cli {
    /// Path to the input .vortex file.
    input: PathBuf,

    /// Path to the output .vortex file.
    output: PathBuf,

    /// Target segment size in megabytes.
    #[arg(long)]
    segment_size_mb: u64,
}

#[cuda_not_available]
fn main() {}

#[cuda_available]
#[tokio::main]
async fn main() -> vortex::error::VortexResult<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt::init();

    let session = VortexSession::default().with_tokio();
    register_cuda_layout(&session);

    let segment_size_bytes = cli.segment_size_mb * 1024 * 1024;

    // Read the input file
    let input_reader = session.open_options().open_path(&cli.input).await?;
    let stream = input_reader.scan()?.into_array_stream()?;

    // Write with the new segment size
    let mut output_file = File::create(&cli.output).await?;
    let write_options = session.write_options().with_strategy(
        WriteStrategyBuilder::default()
            .with_cuda_compatible_encodings()
            .with_coalescing_block_size(segment_size_bytes)
            .build(),
    );

    write_options.write(&mut output_file, stream).await?;
    output_file.flush().await?;

    eprintln!(
        "Wrote {} with {}MB segments",
        cli.output.display(),
        cli.segment_size_mb
    );

    Ok(())
}
