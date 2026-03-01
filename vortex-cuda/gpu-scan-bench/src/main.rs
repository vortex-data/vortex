// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use futures::StreamExt;
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
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PooledFileReadAt;
use vortex_cuda::PooledObjectStoreReadAt;
use vortex_cuda::TracingLaunchStrategy;
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

        let mut registry = tracing_subscriber::registry()
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

        let mut registry = tracing_subscriber::registry()
            .with(log_layer.with_filter(EnvFilter::from_default_env()));

        if let Some(perfetto) = perfetto_guard {
            registry.with(perfetto).init();
        } else {
            registry.init();
        }
    }

    let session = VortexSession::default();
    register_cuda_layout(&session);

    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)?
        .with_launch_strategy(Arc::new(TracingLaunchStrategy));

    let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(
        cuda_ctx.stream().context(),
    )));
    let cuda_stream =
        VortexCudaStreamPool::new(Arc::clone(cuda_ctx.stream().context()), 1).get_stream()?;
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

    for iteration in 0..cli.iterations {
        let start = Instant::now();

        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;

        let mut batches = gpu_file.scan()?.into_array_stream()?;

        let mut chunk = 0;
        while let Some(next) = batches.next().await.transpose()? {
            let len = next.len();
            let span = tracing::info_span!(
                "batch execution",
                iteration = iteration,
                chunk = chunk,
                len = len,
            );

            async {
                next.execute_cuda(&mut cuda_ctx).await?;
                VortexResult::Ok(())
            }
            .instrument(span)
            .await?;

            chunk += 1;
        }

        let elapsed = start.elapsed();
        iteration_times.push(elapsed);
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
    let throughput_mbs = file_size_mb / avg.as_secs_f64();
    let iteration_secs: Vec<f64> = iteration_times.iter().map(|d| d.as_secs_f64()).collect();

    // Always print human-readable to stderr
    eprintln!();
    eprintln!("=== Benchmark Results ===");
    eprintln!("Source:     {}", cli.source);
    eprintln!("Iterations: {}", cli.iterations);
    eprintln!("Avg time:   {:.3}s", avg.as_secs_f64());
    eprintln!("File size:  {file_size_mb:.2} MB");
    eprintln!("Throughput: {throughput_mbs:.2} MB/s");

    Ok(())
}
