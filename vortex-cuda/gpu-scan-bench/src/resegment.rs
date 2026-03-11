// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

//! Reads a CUDA-compatible Vortex file and rewrites it with a different segment size.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use futures::TryStreamExt;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStream;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
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

    /// Repeat the input data this many times in the output file.
    #[arg(long, default_value_t = 1)]
    repeat: usize,
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

    // Read the input file and collect all batches
    let input_reader = session.open_options().open_path(&cli.input).await?;
    let stream = input_reader.scan()?.into_array_stream()?;
    let dtype = stream.dtype().clone();
    let batches: Vec<ArrayRef> = stream.try_collect().await?;

    eprintln!(
        "Read {} batches from {}",
        batches.len(),
        cli.input.display()
    );

    // Build a stream that repeats the batches `repeat` times
    let repeated: Vec<vortex::error::VortexResult<ArrayRef>> = batches
        .iter()
        .cycle()
        .take(batches.len() * cli.repeat)
        .map(|b| Ok(b.clone()))
        .collect();
    let output_stream = ArrayStreamAdapter::new(dtype, futures::stream::iter(repeated));

    // Write with the new segment size
    let mut output_file = File::create(&cli.output).await?;
    let write_options = session.write_options().with_strategy(
        WriteStrategyBuilder::default()
            .with_cuda_compatible_encodings()
            .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
            .with_coalescing_block_size(segment_size_bytes)
            .build(),
    );

    write_options.write(&mut output_file, output_stream).await?;
    output_file.flush().await?;

    eprintln!(
        "Wrote {} with {}MB segments ({}x repeat, {} total batches)",
        cli.output.display(),
        cli.segment_size_mb,
        cli.repeat,
        batches.len() * cli.repeat,
    );

    Ok(())
}
