// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use futures::TryStreamExt;
use futures::stream::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use tracing::Instrument;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::CudaSessionExt;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PooledFileReadAt;
use vortex_cuda::PooledObjectStoreReadAt;
use vortex_cuda::VortexCudaStreamPool;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::register_cuda_layout;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[derive(Parser)]
#[command(
    name = "gpu-scan-bench",
    about = "Benchmark GPU scans of CUDA-compatible Vortex files from S3 or local storage"
)]
struct Cli {
    /// S3 URI (s3://bucket/path) or local path to a CUDA-compatible .vortex file.
    source: String,

    /// Number of scan iterations.
    #[arg(long, default_value_t = 1)]
    iterations: usize,

    /// Path to write Perfetto trace output. If omitted, no trace file is written.
    #[arg(long)]
    perfetto: Option<PathBuf>,

    /// Number of batches to process concurrently (each on its own CUDA stream).
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Skip GPU kernel execution (measure IO + deserialization only).
    #[arg(long)]
    no_execute: bool,

    /// Output logs as JSON.
    #[arg(long)]
    json: bool,
}

#[cuda_not_available]
fn main() {}

#[cuda_available]
#[tokio::main]
async fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    // Setup tracing
    let perfetto_guard = if let Some(ref perfetto_path) = cli.perfetto {
        let perfetto_file = File::create(perfetto_path)?;
        Some(PerfettoLayer::new(perfetto_file).with_debug_annotations(true))
    } else {
        None
    };

    if cli.json {
        let log_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false);

        let registry = tracing_subscriber::registry()
            .with(log_layer.with_filter(EnvFilter::from_default_env()));

        if let Some(perfetto) = perfetto_guard {
            registry.with(perfetto).init();
        } else {
            registry.init();
        }
    } else {
        let log_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false)
            .event_format(tracing_subscriber::fmt::format().with_target(true));

        let registry = tracing_subscriber::registry()
            .with(log_layer.with_filter(EnvFilter::from_default_env()));

        if let Some(perfetto) = perfetto_guard {
            registry.with(perfetto).init();
        } else {
            registry.init();
        }
    }

    let session = VortexSession::default().with_tokio();
    register_cuda_layout(&session);

    let cuda_context = session.cuda_session().context().clone();

    let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(&cuda_context)));
    let cuda_stream = VortexCudaStreamPool::new(Arc::clone(&cuda_context), 1).get_stream()?;
    let handle = session.handle();

    // Parse source and create reader
    let reader: Arc<dyn vortex::io::VortexReadAt> = if cli.source.starts_with("s3://") {
        let url = Url::parse(&cli.source)
            .map_err(|e| vortex::error::vortex_err!("invalid S3 URL: {e}"))?;
        let bucket = url
            .host_str()
            .ok_or_else(|| vortex::error::vortex_err!("S3 URL missing bucket name"))?;
        let path = ObjectPath::from(url.path());
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?,
        );
        Arc::new(PooledObjectStoreReadAt::new(
            store,
            path,
            handle,
            Arc::clone(&pool),
            cuda_stream,
        ))
    } else {
        let path = PathBuf::from(&cli.source);
        Arc::new(PooledFileReadAt::open(
            &path,
            handle,
            Arc::clone(&pool),
            cuda_stream,
        )?)
    };

    // Run benchmark iterations
    let mut iteration_times = Vec::with_capacity(cli.iterations);
    let mut output_bytes: u64 = 0;
    let concurrency = cli.concurrency;
    let no_execute = cli.no_execute;

    for iteration in 0..cli.iterations {
        let start = Instant::now();

        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;

        let batches = gpu_file.scan()?.into_array_stream()?;

        let batch_bytes: Vec<u64> = batches
            .enumerate()
            .map(|(chunk, batch)| {
                let session = &session;
                async move {
                    let batch = batch?;
                    let len = batch.len();
                    let span = tracing::info_span!(
                        "batch execution",
                        iteration = iteration,
                        chunk = chunk,
                        len = len,
                    );

                    async {
                        if no_execute {
                            tracing::info!(len, "skipping execute (--no-execute)");
                            VortexResult::Ok(0u64)
                        } else {
                            let mut cuda_ctx = CudaSession::create_execution_ctx(session)?;
                            let canonical = batch.execute_cuda(&mut cuda_ctx).await?;
                            let nbytes = canonical.into_array().nbytes();
                            VortexResult::Ok(nbytes)
                        }
                    }
                    .instrument(span)
                    .await
                }
            })
            .buffered(concurrency)
            .try_collect()
            .await?;

        let elapsed = start.elapsed();
        iteration_times.push(elapsed);
        if iteration == 0 {
            output_bytes = batch_bytes.iter().sum();
        }
        tracing::info!(
            "Iteration {}/{}: {:.3}s",
            iteration + 1,
            cli.iterations,
            elapsed.as_secs_f64()
        );
    }

    // Compute summary stats
    let total: std::time::Duration = iteration_times.iter().sum();
    let avg = total / iteration_times.len() as u32;
    let file_size = reader.size().await?;
    let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
    let output_size_mb = output_bytes as f64 / (1024.0 * 1024.0);
    let input_throughput_mbs = file_size_mb / avg.as_secs_f64();
    let output_throughput_mbs = output_size_mb / avg.as_secs_f64();
    // Always print human-readable to stderr
    eprintln!();
    eprintln!("=== Benchmark Results ===");
    eprintln!("Source:      {}", cli.source);
    eprintln!("Iterations:  {}", cli.iterations);
    eprintln!("Avg time:    {:.3}s", avg.as_secs_f64());
    eprintln!("Input size:  {file_size_mb:.2} MB");
    eprintln!("Output size: {output_size_mb:.2} MB");
    eprintln!("Input throughput:  {input_throughput_mbs:.2} MB/s");
    eprintln!("Output throughput: {output_throughput_mbs:.2} MB/s");

    Ok(())
}
