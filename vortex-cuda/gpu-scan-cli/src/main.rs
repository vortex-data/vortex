// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use clap::Subcommand;
use futures::StreamExt;
use tracing::Instrument;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use vortex::VortexSessionDefault;
#[expect(deprecated)]
use vortex::array::ToCanonical as _;
use vortex::array::arrays::Dict;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::buffer::ByteBufferMut;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PooledByteBufferReadAt;
use vortex_cuda::PooledFileReadAt;
use vortex_cuda::TracingLaunchStrategy;
use vortex_cuda::VortexCudaStreamPool;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
use vortex_cuda::layout::register_cuda_layout;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[derive(Parser)]
#[command(name = "gpu-scan-cli", about = "CUDA GPU scan tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a Vortex file to use CUDA-compatible encodings.
    Convert {
        /// Path to input .vortex file.
        #[arg(long)]
        input: PathBuf,
        /// Path to output CUDA-compatible .vortex file.
        #[arg(long)]
        output: PathBuf,
    },
    /// Scan a Vortex file using GPU decompression.
    Scan {
        /// Path to .vortex file.
        path: PathBuf,
        /// If set, the file is already CUDA-compatible (skip recompression).
        #[arg(long)]
        gpu_file: bool,
        /// Output logs as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[cuda_not_available]
fn main() {}

#[cuda_available]
#[tokio::main]
async fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Convert { input, output } => cmd_convert(input, output).await,
        Command::Scan {
            path,
            gpu_file,
            json,
        } => cmd_scan(path, gpu_file, json).await,
    }
}

/// Build the write strategy used for CUDA-compatible file output.
#[cuda_available]
fn cuda_write_strategy() -> Arc<dyn vortex::layout::LayoutStrategy> {
    WriteStrategyBuilder::default()
        .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().only_cuda_compatible())
        .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
        .build()
}

/// Convert an input Vortex file to CUDA-compatible encodings and write to disk.
#[cuda_available]
async fn cmd_convert(input: PathBuf, output: PathBuf) -> VortexResult<()> {
    let session = VortexSession::default();
    register_cuda_layout(&session);

    let input_file = session.open_options().open_path(&input).await?;
    let scan = input_file.scan()?.into_array_stream()?;

    let mut out = tokio::fs::File::create(&output).await?;
    session
        .write_options()
        .with_strategy(cuda_write_strategy())
        .write(&mut out, scan)
        .await?;

    tracing::info!("Wrote CUDA-compatible file to {}", output.display());
    Ok(())
}

/// Scan a Vortex file on the GPU, optionally recompressing in memory first.
#[cuda_available]
async fn cmd_scan(path: PathBuf, gpu_file: bool, json_output: bool) -> VortexResult<()> {
    let perfetto_file = File::create("trace.pb")?;
    let perfetto_layer = PerfettoLayer::new(perfetto_file).with_debug_annotations(true);

    if json_output {
        let log_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false);

        tracing_subscriber::registry()
            .with(perfetto_layer)
            .with(log_layer.with_filter(EnvFilter::from_default_env()))
            .init();
    } else {
        let log_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false)
            .event_format(tracing_subscriber::fmt::format().with_target(true));

        tracing_subscriber::registry()
            .with(perfetto_layer)
            .with(log_layer.with_filter(EnvFilter::from_default_env()))
            .init();
    }

    let session = VortexSession::default();
    register_cuda_layout(&session);

    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)?
        .with_launch_strategy(Arc::new(TracingLaunchStrategy));

    let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(
        cuda_ctx.stream().context(),
    )));
    let cuda_stream =
        VortexCudaStreamPool::new(Arc::clone(cuda_ctx.stream().context()), 1).stream()?;
    let handle = session.handle();

    let gpu_file_handle = if gpu_file {
        let reader = PooledFileReadAt::open(&path, handle, Arc::clone(&pool), cuda_stream)?;
        session.open_options().open(Arc::new(reader)).await?
    } else {
        let (recompressed, footer) = recompress_for_gpu(&path, &session).await?;
        let reader = PooledByteBufferReadAt::new(recompressed, Arc::clone(&pool), cuda_stream);
        session
            .open_options()
            .with_footer(footer)
            .open(Arc::new(reader))
            .await?
    };

    let mut batches = gpu_file_handle.scan()?.into_array_stream()?;

    let mut chunk = 0;
    while let Some(next) = batches.next().await.transpose()? {
        #[expect(deprecated)]
        let record = next.to_struct();

        for (field, field_name) in record
            .iter_unmasked_fields()
            .zip(record.struct_fields().names().iter())
        {
            let field_name = field_name.to_string();
            if field.is::<Dict>() {
                continue;
            }

            let len = field.len();

            let span = tracing::info_span!(
                "array execution",
                chunk = chunk,
                field_name = field_name,
                len = len,
            );

            async {
                if field.clone().execute_cuda(&mut cuda_ctx).await.is_err() {
                    tracing::error!("failed to execute_cuda on column");
                }
            }
            .instrument(span)
            .await;
        }

        chunk += 1;
    }

    Ok(())
}

/// Recompress the input file using CUDA-compatible encodings, returning the file as an
/// in-memory byte buffer along with its footer.
#[cuda_available]
async fn recompress_for_gpu(
    input_path: impl AsRef<std::path::Path>,
    session: &VortexSession,
) -> VortexResult<(vortex::buffer::ByteBuffer, vortex::file::Footer)> {
    let input = session.open_options().open_path(input_path).await?;
    let scan = input.scan()?.into_array_stream()?;

    let mut out = ByteBufferMut::empty();
    let result = session
        .write_options()
        .with_strategy(cuda_write_strategy())
        .write(&mut out, scan)
        .await?;

    Ok((out.freeze(), result.footer().clone()))
}
