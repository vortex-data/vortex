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
use vortex::io::CoalesceConfig;
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

    /// Scan concurrency (splits per worker thread).
    #[arg(long, default_value_t = 4)]
    scan_concurrency: usize,

    /// Number of CUDA streams for H2D transfers (round-robin).
    #[arg(long, default_value_t = 1)]
    cuda_streams: usize,

    /// Override IO driver concurrency (max concurrent read_at calls).
    /// Defaults to 192 for S3, 32 for local files.
    #[arg(long)]
    io_concurrency: Option<usize>,

    /// Override coalesce max request size in MB. Default: 16 for S3, 4 for files.
    #[arg(long)]
    coalesce_max_mb: Option<u64>,

    /// Override coalesce distance in KB. Default: 1024 (1MB).
    #[arg(long)]
    coalesce_distance_kb: Option<u64>,

    /// Disable IO request coalescing entirely.
    #[arg(long)]
    no_coalesce: bool,

    /// Known compression ratio (input/output). Skips the extra scan pass to measure output size.
    #[arg(long)]
    compression_ratio: Option<f64>,
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
    let cuda_stream_pool = VortexCudaStreamPool::new(Arc::clone(&cuda_context), cli.cuda_streams);
    let cuda_streams: Vec<_> = (0..cli.cuda_streams.max(1))
        .map(|_| cuda_stream_pool.get_stream())
        .collect::<VortexResult<_>>()?;
    let handle = session.handle();

    // Build coalesce config override
    let coalesce_override: Option<Option<CoalesceConfig>> = if cli.no_coalesce {
        Some(None)
    } else if cli.coalesce_max_mb.is_some() || cli.coalesce_distance_kb.is_some() {
        Some(Some(CoalesceConfig::new(
            cli.coalesce_distance_kb.unwrap_or(1024) * 1024,
            cli.coalesce_max_mb.unwrap_or(16) * 1024 * 1024,
        )))
    } else {
        None
    };

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
        let mut s3_reader = PooledObjectStoreReadAt::new_with_streams(
            store,
            path,
            handle,
            Arc::clone(&pool),
            cuda_streams,
        );
        if let Some(io_concurrency) = cli.io_concurrency {
            s3_reader = s3_reader.with_concurrency(io_concurrency);
        }
        if let Some(coalesce) = coalesce_override {
            s3_reader = s3_reader.with_some_coalesce_config(coalesce);
        }
        Arc::new(s3_reader)
    } else {
        let path = PathBuf::from(&cli.source);
        Arc::new(PooledFileReadAt::open_with_streams(
            &path,
            handle,
            Arc::clone(&pool),
            cuda_streams,
        )?)
    };

    // Run benchmark iterations
    let mut iteration_times = Vec::with_capacity(cli.iterations);
    let concurrency = cli.concurrency;
    let no_execute = cli.no_execute;

    for iteration in 0..cli.iterations {
        let start = Instant::now();

        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;

        let batches = gpu_file
            .scan()?
            .with_concurrency(cli.scan_concurrency)
            .into_array_stream()?;

        batches
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
                        } else {
                            let mut cuda_ctx = CudaSession::create_execution_ctx(session)?;
                            batch.execute_cuda(&mut cuda_ctx).await?;
                        }
                        VortexResult::Ok(())
                    }
                    .instrument(span)
                    .await
                }
            })
            .buffered(concurrency)
            .try_collect::<Vec<_>>()
            .await?;

        let elapsed = start.elapsed();
        iteration_times.push(elapsed);
        tracing::info!(
            "Iteration {}/{}: {:.3}s",
            iteration + 1,
            cli.iterations,
            elapsed.as_secs_f64()
        );
    }

    // Measure output size: use compression ratio if provided, otherwise run a separate pass
    let file_size = reader.size().await?;
    let output_bytes: u64 = if let Some(ratio) = cli.compression_ratio {
        (file_size as f64 * ratio) as u64
    } else if !no_execute {
        let gpu_file = session.open_options().open(Arc::clone(&reader)).await?;
        let batches = gpu_file
            .scan()?
            .with_concurrency(cli.scan_concurrency)
            .into_array_stream()?;
        batches
            .map(|batch| {
                let session = &session;
                async move {
                    let batch = batch?;
                    let mut cuda_ctx = CudaSession::create_execution_ctx(session)?;
                    let canonical = batch.execute_cuda(&mut cuda_ctx).await?;
                    VortexResult::Ok(canonical.into_array().nbytes())
                }
            })
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await?
            .iter()
            .sum()
    } else {
        0
    };

    // Compute summary stats
    let total: std::time::Duration = iteration_times.iter().sum();
    let avg = total / iteration_times.len() as u32;
    let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
    let output_size_mb = output_bytes as f64 / (1024.0 * 1024.0);
    let input_throughput_mbs = file_size_mb / avg.as_secs_f64();
    let output_throughput_mbs = output_size_mb / avg.as_secs_f64();
    // Always print human-readable to stderr
    eprintln!();
    eprintln!("=== Benchmark Results ===");
    eprintln!("Source:           {}", cli.source);
    eprintln!("Iterations:       {}", cli.iterations);
    eprintln!("Scan concurrency: {} per thread", cli.scan_concurrency);
    eprintln!("CUDA streams:     {}", cli.cuda_streams);
    if let Some(io_c) = cli.io_concurrency {
        eprintln!("IO concurrency:   {io_c} (override)");
    }
    if cli.no_coalesce {
        eprintln!("Coalescing:       disabled");
    } else if let Some(ref cc) = coalesce_override {
        if let Some(cc) = cc {
            eprintln!(
                "Coalesce config:  distance={}KB, max={}MB (override)",
                cc.distance / 1024,
                cc.max_size / (1024 * 1024)
            );
        }
    }
    eprintln!("Avg time:    {:.3}s", avg.as_secs_f64());
    eprintln!("Input size:  {file_size_mb:.2} MB");
    eprintln!("Output size: {output_size_mb:.2} MB");
    eprintln!("Input throughput:  {input_throughput_mbs:.2} MB/s");
    eprintln!("Output throughput: {output_throughput_mbs:.2} MB/s");

    Ok(())
}
