// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::str::FromStr;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
#[cfg(target_os = "linux")]
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use cudarc::driver::DevicePtr;
use cudarc::driver::result;
use cudarc::driver::sys;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
#[cfg(target_os = "linux")]
use ktls::CorkStream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::ObjectStore;
use object_store::ObjectStoreScheme;
use object_store::aws::AmazonS3;
use object_store::aws::AmazonS3Builder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectStorePath;
#[cfg(target_os = "linux")]
use object_store::signer::Signer;
use parking_lot::Mutex;
#[cfg(target_os = "linux")]
use rustls::ClientConfig;
#[cfg(target_os = "linux")]
use rustls::RootCertStore;
#[cfg(target_os = "linux")]
use rustls::pki_types::ServerName;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::Notify;
#[cfg(target_os = "linux")]
use tokio_rustls::TlsConnector;
use tracing_subscriber::EnvFilter;
use url::Url;
use vortex::array::Array;
use vortex::array::MaskFuture;
use vortex::array::buffer::BufferHandle;
use vortex::array::expr::Expression;
use vortex::array::expr::col;
use vortex::array::expr::eq;
use vortex::array::expr::gt;
use vortex::array::expr::gt_eq;
use vortex::array::expr::is_root;
use vortex::array::expr::lit;
use vortex::array::expr::lt;
use vortex::array::expr::lt_eq;
use vortex::array::expr::not_eq;
use vortex::array::expr::root;
use vortex::array::expr::select;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::segments::io_request_stats;
use vortex::file::segments::reset_io_request_stats;
use vortex::io::BufferAllocator;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::WriteTarget;
use vortex::io::copy_stats;
use vortex::io::default_alloc_stats;
use vortex::io::reset_copy_stats;
use vortex::io::reset_default_alloc_stats;
use vortex::layout::LayoutReader;
use vortex::mask::Mask;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::MetricsRegistry;
use vortex_bench::SESSION;
use vortex_bench::rdma_proto::DEFAULT_RDMA_PORT;
use vortex_bench::rdma_proto::OP_IPC_HANDLE;
use vortex_bench::rdma_proto::OP_LIST;
use vortex_bench::rdma_proto::OP_READ;
use vortex_bench::rdma_proto::OP_SIZE;
use vortex_bench::rdma_proto::read_status;
use vortex_bench::rdma_proto::read_string;
use vortex_bench::rdma_proto::write_string;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaSessionExt;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PinnedDeviceAllocator;
use vortex_scan::ScanBuilder;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Benchmark Vortex scans over local files vs object stores"
)]
struct Args {
    /// File path, directory, or URL (e.g. file:/..., s3://bucket/path, rdma://host:9900/path, p2p://host:9900/path)
    #[arg(long)]
    source: String,
    /// Use object_store even for file: URLs
    #[arg(long, default_value_t = false)]
    force_object_store: bool,
    /// Run a predefined scan shape.
    #[arg(long, value_enum)]
    preset: Option<Preset>,
    /// Projection field names (comma-separated).
    #[arg(long, value_delimiter = ',')]
    projection: Option<Vec<String>>,
    /// Filter column name.
    #[arg(long)]
    filter_col: Option<String>,
    /// Filter operator.
    #[arg(long, value_enum)]
    filter_op: Option<FilterOp>,
    /// Filter literal value (integer).
    #[arg(long)]
    filter_value: Option<i64>,
    /// Filter literal type.
    #[arg(long, value_enum, default_value_t = LiteralType::I64)]
    filter_type: LiteralType,
    /// Number of scan iterations.
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    /// Scan concurrency (tasks per thread).
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
    /// Max files scanned in parallel (file-level readahead).
    #[arg(long, default_value_t = 1)]
    file_concurrency: usize,
    /// Reopen the file for each iteration to avoid caching effects.
    #[arg(long, default_value_t = false)]
    reopen: bool,
    /// Which scan path to use.
    #[arg(long, value_enum, default_value_t = ScanMode::Full)]
    mode: ScanMode,
    /// Only read segments and drop buffers (skip decode/projection).
    #[arg(long, default_value_t = false)]
    io_only: bool,
    /// Only prune whole segments (no intra-segment pruning on CPU).
    #[arg(long, default_value_t = false)]
    prune_segments: bool,
    /// Enable CUDA pinned read + H2D transfer.
    #[arg(long, default_value_t = false)]
    gpu: bool,
    /// Number of CUDA streams for H2D transfers (requires --gpu).
    #[arg(long, default_value_t = 4)]
    gpu_streams: usize,
}

#[derive(ValueEnum, Clone, Debug)]
enum ScanMode {
    /// Read segments only (no decode).
    Io,
    /// Decode arrays without filter evaluation.
    Decode,
    /// Decode arrays with full filter/projection evaluation.
    Full,
}

#[derive(ValueEnum, Clone, Debug)]
enum Preset {
    /// ClickBench query #2: AdvEngineID != 0, projecting AdvEngineID.
    Clickbench2,
}

#[derive(ValueEnum, Clone, Debug)]
enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
}

#[derive(ValueEnum, Clone, Debug, Copy)]
enum LiteralType {
    I16,
    I32,
    I64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let mode = if args.io_only {
        ScanMode::Io
    } else {
        args.mode.clone()
    };
    let source_is_p2p = Url::parse(&args.source)
        .ok()
        .is_some_and(|url| url.scheme() == "p2p");
    // `vortex-cuda` datasets require this layout decoder even when running
    // CPU decode/full scans (i.e. without `--gpu` data movement).
    vortex_cuda::layout::register_cuda_layout(&SESSION);

    let (projection, filter) = build_scan_exprs(&args)?;
    let metrics: std::sync::Arc<dyn MetricsRegistry> =
        std::sync::Arc::new(DefaultMetricsRegistry::default());
    let use_gpu_allocator = args.gpu && !source_is_p2p;
    #[allow(clippy::if_then_some_else_none)]
    let (gpu_allocator, pinned_pool) = if use_gpu_allocator {
        let cuda_session = SESSION.cuda_session();
        let pool = std::sync::Arc::new(PinnedByteBufferPool::new(cuda_session.context().clone()));
        let allocator = std::sync::Arc::new(PinnedDeviceAllocator::from_session_with_streams(
            pool.clone(),
            &SESSION,
            args.gpu_streams,
        )?);
        (Some(allocator), Some(pool))
    } else {
        (None, None)
    };
    let allocator: Option<std::sync::Arc<dyn BufferAllocator>> = gpu_allocator
        .as_ref()
        .map(|alloc| alloc.clone() as std::sync::Arc<dyn BufferAllocator>);

    let targets = resolve_targets(&args).await?;
    let all_files = open_all_targets(
        &targets,
        metrics.clone(),
        args.file_concurrency,
        allocator.clone(),
    )
    .await?;
    let data_size_per_iter: u64 = all_files
        .iter()
        .map(|f| {
            f.footer()
                .segment_map()
                .iter()
                .map(|s| s.length as u64)
                .sum::<u64>()
        })
        .sum();
    let cached_files = if args.reopen {
        None
    } else {
        Some(std::sync::Arc::new(all_files))
    };
    reset_default_alloc_stats();
    reset_copy_stats();
    reset_io_request_stats();
    if let Some(pool) = pinned_pool.as_ref() {
        pool.reset_stats();
    }

    let start = Instant::now();
    let first_seen = std::sync::Arc::new(AtomicBool::new(false));
    let first_latency = std::sync::Arc::new(Mutex::new(None::<f64>));
    let targets = targets.clone();

    let rows = futures::stream::iter(0..args.iterations)
        .flat_map(|_| futures::stream::iter(targets.clone().into_iter().enumerate()))
        .map(|(idx, target)| {
            let cached_files = cached_files.clone();
            let projection = projection.clone();
            let filter = filter.clone();
            let metrics = metrics.clone();
            let first_seen = first_seen.clone();
            let first_latency = first_latency.clone();
            let mode = mode.clone();
            let allocator = allocator.clone();
            async move {
                let file = match &cached_files {
                    Some(files) => files[idx].clone(),
                    None => {
                        open_vortex_file_for_target(&target, metrics.clone(), allocator).await?
                    }
                };

                if args.prune_segments
                    && let Some(filter) = filter.as_ref()
                    && file.can_prune(filter)?
                {
                    return Ok::<_, anyhow::Error>(0);
                }

                if matches!(mode, ScanMode::Io) {
                    if io_no_copy_enabled() {
                        match &target {
                            ScanTarget::ObjectStore { store, path } => {
                                read_all_segments_object_store_no_copy(
                                    &file,
                                    store,
                                    path,
                                    args.concurrency,
                                )
                                .await?;
                            }
                            ScanTarget::AmazonS3 { store, path } => {
                                let store_dyn: std::sync::Arc<dyn ObjectStore> = store.clone();
                                read_all_segments_object_store_no_copy(
                                    &file,
                                    &store_dyn,
                                    path,
                                    args.concurrency,
                                )
                                .await?;
                            }
                            ScanTarget::Local(_) => {
                                read_all_segments(&file, args.concurrency).await?;
                            }
                            ScanTarget::Rdma { .. } => {
                                read_all_segments(&file, args.concurrency).await?;
                            }
                            ScanTarget::P2p { .. } => {
                                read_all_segments(&file, args.concurrency).await?;
                            }
                        }
                    } else {
                        read_all_segments(&file, args.concurrency).await?;
                    }
                    if !first_seen.load(Ordering::Relaxed)
                        && !first_seen.swap(true, Ordering::Relaxed)
                    {
                        *first_latency.lock() = Some(start.elapsed().as_secs_f64());
                    }
                    let file_rows = usize::try_from(file.row_count())
                        .map_err(|_| anyhow::anyhow!("row_count exceeds usize"))?;
                    drop(file);
                    return Ok::<_, anyhow::Error>(file_rows);
                }

                let (scan_projection, scan_filter, bypass_filter) = match mode {
                    ScanMode::Decode => {
                        let scan_filter = if args.prune_segments {
                            filter.clone()
                        } else {
                            None
                        };
                        (root(), scan_filter, true)
                    }
                    ScanMode::Full => (projection.clone(), filter.clone(), false),
                    ScanMode::Io => unreachable!("io-only handled above"),
                };

                if should_use_direct_root_scan(&mode, &scan_projection, scan_filter.as_ref(), args.prune_segments)
                {
                    let file_rows = direct_root_scan(&file).await?;
                    if !first_seen.load(Ordering::Relaxed)
                        && !first_seen.swap(true, Ordering::Relaxed)
                    {
                        *first_latency.lock() = Some(start.elapsed().as_secs_f64());
                    }
                    drop(file);
                    return Ok::<_, anyhow::Error>(file_rows);
                }

                let layout_reader = file.layout_reader()?;
                let layout_reader = if args.prune_segments || bypass_filter {
                    std::sync::Arc::new(BenchLayoutReader::new(
                        layout_reader,
                        args.prune_segments,
                        bypass_filter,
                    )) as std::sync::Arc<dyn LayoutReader>
                } else {
                    layout_reader
                };

                let scan = ScanBuilder::new(SESSION.clone(), layout_reader)
                    .with_metrics_registry(metrics.clone())
                    .with_projection(scan_projection)
                    .with_some_filter(scan_filter)
                    .with_concurrency(args.concurrency)
                    .map(|array| Ok(array.len()));

                let mut stream = scan.into_stream()?;
                let mut file_rows = 0usize;
                while let Some(rows) = stream.try_next().await? {
                    if !first_seen.load(Ordering::Relaxed)
                        && !first_seen.swap(true, Ordering::Relaxed)
                    {
                        *first_latency.lock() = Some(start.elapsed().as_secs_f64());
                    }
                    file_rows += rows;
                }

                drop(file);
                Ok::<_, anyhow::Error>(file_rows)
            }
        })
        .buffer_unordered(args.file_concurrency.max(1))
        .try_fold(
            0usize,
            |rows, file_rows| async move { Ok(rows + file_rows) },
        )
        .await?;

    let elapsed = start.elapsed().as_secs_f64();
    let allocator_sync_ms = if let Some(allocator) = gpu_allocator {
        let sync_start = Instant::now();
        allocator.synchronize()?;
        sync_start.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let p2p_sync_ms = synchronize_registered_p2p_streams()?;
    let gpu_sync_ms = allocator_sync_ms + p2p_sync_ms;
    let first_latency_s = (*first_latency.lock()).unwrap_or(elapsed);
    let avg_time_s = elapsed / args.iterations as f64;
    let data_mb = data_size_per_iter as f64 / (1024.0 * 1024.0);
    let avg_mb_s = if avg_time_s > 0.0 {
        data_mb / avg_time_s
    } else {
        0.0
    };
    let steady_time_s =
        (elapsed - first_latency_s).max(0.0) / (args.iterations as f64 - 1.0).max(1.0);
    let steady_mb_s = if steady_time_s > 0.0 {
        data_mb / steady_time_s
    } else {
        0.0
    };

    println!("files={}", targets.len());
    println!("rows={}", rows / args.iterations);
    println!("data_size_mb={:.2}", data_mb);
    println!("avg_time_s={:.3}", avg_time_s);
    println!("avg_mb_s={:.2}", avg_mb_s);
    println!("first_latency_ms={:.2}", first_latency_s * 1000.0);
    println!("steady_mb_s={:.2}", steady_mb_s);
    if io_no_copy_enabled() {
        println!("io_no_copy=1");
        println!(
            "io_no_copy_coalesce_distance={} io_no_copy_coalesce_max_size={}",
            no_copy_coalesce_distance(),
            no_copy_coalesce_max_size()
        );
    }
    if s3_ktls_enabled() {
        println!("s3_ktls=1");
    }
    if args.gpu || p2p_sync_ms > 0.0 {
        println!("gpu_sync_ms={:.2}", gpu_sync_ms);
    }
    print_stats(pinned_pool.as_deref());

    Ok(())
}

fn build_scan_exprs(args: &Args) -> VortexResult<(Expression, Option<Expression>)> {
    if let Some(preset) = &args.preset {
        return build_preset_exprs(preset);
    }

    let projection = match &args.projection {
        Some(fields) if !fields.is_empty() => {
            let names = FieldNames::from_iter(fields.iter().map(|s| s.as_str()));
            select(names, root())
        }
        _ => root(),
    };

    let filter = match (&args.filter_col, &args.filter_op, args.filter_value) {
        (Some(col_name), Some(op), Some(value)) => {
            let lhs = col(col_name.as_str());
            let rhs = match args.filter_type {
                LiteralType::I16 => lit(i16::try_from(value)
                    .map_err(|_| vortex_err!("filter_value does not fit in i16"))?),
                LiteralType::I32 => lit(i32::try_from(value)
                    .map_err(|_| vortex_err!("filter_value does not fit in i32"))?),
                LiteralType::I64 => lit(value),
            };
            Some(apply_filter_op(op.clone(), lhs, rhs))
        }
        _ => None,
    };

    Ok((projection, filter))
}

fn should_use_direct_root_scan(
    mode: &ScanMode,
    projection: &Expression,
    filter: Option<&Expression>,
    prune_segments: bool,
) -> bool {
    matches!(mode, ScanMode::Full | ScanMode::Decode)
        && filter.is_none()
        && !prune_segments
        && is_root(projection)
}

async fn direct_root_scan(file: &vortex::file::VortexFile) -> Result<usize> {
    let row_count = file.row_count();
    let row_len = usize::try_from(row_count)
        .map_err(|_| anyhow::anyhow!("row_count exceeds usize"))?;
    let layout_reader = file.layout_reader()?;
    let array = layout_reader
        .projection_evaluation(&(0..row_count), &root(), MaskFuture::new_true(row_len))?
        .await?;
    Ok(array.len())
}

fn build_preset_exprs(preset: &Preset) -> VortexResult<(Expression, Option<Expression>)> {
    match preset {
        Preset::Clickbench2 => {
            let projection = select(["AdvEngineID"], root());
            let filter = not_eq(col("AdvEngineID"), lit(0_i16));
            Ok((projection, Some(filter)))
        }
    }
}

fn apply_filter_op(op: FilterOp, lhs: Expression, rhs: Expression) -> Expression {
    match op {
        FilterOp::Eq => eq(lhs, rhs),
        FilterOp::Neq => not_eq(lhs, rhs),
        FilterOp::Gt => gt(lhs, rhs),
        FilterOp::Gte => gt_eq(lhs, rhs),
        FilterOp::Lt => lt(lhs, rhs),
        FilterOp::Lte => lt_eq(lhs, rhs),
    }
}

#[derive(Clone)]
enum ScanTarget {
    Local(PathBuf),
    Rdma {
        endpoint: std::sync::Arc<str>,
        key: std::sync::Arc<str>,
    },
    P2p {
        endpoint: std::sync::Arc<str>,
        key: std::sync::Arc<str>,
    },
    AmazonS3 {
        store: std::sync::Arc<AmazonS3>,
        path: ObjectStorePath,
    },
    ObjectStore {
        store: std::sync::Arc<dyn ObjectStore>,
        path: ObjectStorePath,
    },
}

#[derive(Clone)]
enum ResolvedStore {
    AmazonS3(std::sync::Arc<AmazonS3>),
    ObjectStore(std::sync::Arc<dyn ObjectStore>),
}

async fn resolve_targets(args: &Args) -> Result<Vec<ScanTarget>> {
    let source = &args.source;

    if let Ok(url) = Url::parse(source) {
        if url.scheme() == "rdma" {
            let (endpoint, key_or_prefix) = rdma_endpoint_and_path(&url)?;
            if is_prefix(source) {
                let objects = rdma_list(&endpoint, &key_or_prefix).await?;
                let mut targets = objects
                    .into_iter()
                    .map(|obj| ScanTarget::Rdma {
                        endpoint: std::sync::Arc::from(endpoint.clone()),
                        key: std::sync::Arc::from(obj.key),
                    })
                    .collect::<Vec<_>>();
                targets.sort_by(|a, b| {
                    let ScanTarget::Rdma { key: ka, .. } = a else {
                        unreachable!("Rdma-only target list expected")
                    };
                    let ScanTarget::Rdma { key: kb, .. } = b else {
                        unreachable!("Rdma-only target list expected")
                    };
                    ka.cmp(kb)
                });
                return Ok(targets);
            }

            return Ok(vec![ScanTarget::Rdma {
                endpoint: std::sync::Arc::from(endpoint),
                key: std::sync::Arc::from(key_or_prefix),
            }]);
        }
        if url.scheme() == "p2p" {
            let (endpoint, key_or_prefix) = p2p_endpoint_and_path(&url)?;
            if is_prefix(source) {
                let objects = rdma_list(&endpoint, &key_or_prefix).await?;
                let mut targets = objects
                    .into_iter()
                    .map(|obj| ScanTarget::P2p {
                        endpoint: std::sync::Arc::from(endpoint.clone()),
                        key: std::sync::Arc::from(obj.key),
                    })
                    .collect::<Vec<_>>();
                targets.sort_by(|a, b| {
                    let ScanTarget::P2p { key: ka, .. } = a else {
                        unreachable!("P2p-only target list expected")
                    };
                    let ScanTarget::P2p { key: kb, .. } = b else {
                        unreachable!("P2p-only target list expected")
                    };
                    ka.cmp(kb)
                });
                return Ok(targets);
            }

            return Ok(vec![ScanTarget::P2p {
                endpoint: std::sync::Arc::from(endpoint),
                key: std::sync::Arc::from(key_or_prefix),
            }]);
        }

        if url.scheme() == "file" && !args.force_object_store {
            let path = url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL: {source}"))?;
            return Ok(resolve_local_targets(&path));
        }

        let (scheme, store, path) = object_store_from_url(source)?;
        if is_prefix(source) {
            if matches!(scheme, ObjectStoreScheme::Http) {
                anyhow::bail!("HTTP object stores do not support listing prefixes");
            }

            let mut targets = Vec::new();
            match &store {
                ResolvedStore::AmazonS3(s3) => {
                    let store_dyn: std::sync::Arc<dyn ObjectStore> = s3.clone();
                    let mut entries = store_dyn.list(Some(&path));
                    while let Some(entry) = entries.try_next().await? {
                        targets.push(ScanTarget::AmazonS3 {
                            store: s3.clone(),
                            path: entry.location.clone(),
                        });
                    }
                }
                ResolvedStore::ObjectStore(store) => {
                    let mut entries = store.list(Some(&path));
                    while let Some(entry) = entries.try_next().await? {
                        targets.push(ScanTarget::ObjectStore {
                            store: store.clone(),
                            path: entry.location.clone(),
                        });
                    }
                }
            }
            return Ok(targets);
        }

        return Ok(vec![match store {
            ResolvedStore::AmazonS3(store) => ScanTarget::AmazonS3 { store, path },
            ResolvedStore::ObjectStore(store) => ScanTarget::ObjectStore { store, path },
        }]);
    }

    let path = PathBuf::from(source);
    Ok(resolve_local_targets(&path))
}

fn resolve_local_targets(path: &Path) -> Vec<ScanTarget> {
    if path.is_dir() {
        let mut entries = match std::fs::read_dir(path) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|entry| entry.extension().is_some_and(|e| e == "vortex"))
                .collect::<Vec<_>>(),
            Err(_) => Vec::new(),
        };
        entries.sort();
        entries.into_iter().map(ScanTarget::Local).collect()
    } else {
        vec![ScanTarget::Local(path.to_path_buf())]
    }
}

fn is_prefix(source: &str) -> bool {
    source.ends_with('/')
}

async fn open_vortex_file_for_target(
    target: &ScanTarget,
    metrics: std::sync::Arc<dyn MetricsRegistry>,
    allocator: Option<std::sync::Arc<dyn BufferAllocator>>,
) -> Result<vortex::file::VortexFile> {
    let session = SESSION.clone();
    match target {
        ScanTarget::Local(path) => {
            let mut options = session.open_options().with_metrics_registry(metrics);
            if let Some(allocator) = allocator {
                options = options.with_allocator(allocator);
            }
            Ok(options.open_path(path).await?)
        }
        ScanTarget::Rdma { endpoint, key } => {
            let mut options = session.open_options().with_metrics_registry(metrics);
            if let Some(allocator) = allocator {
                options = options.with_allocator(allocator);
            }
            let source = std::sync::Arc::new(
                RdmaReadSource::new(endpoint.clone(), key.clone())
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to create rdma source: {e}"))?,
            );
            Ok(options.open(source).await?)
        }
        ScanTarget::P2p { endpoint, key } => {
            let options = session.open_options().with_metrics_registry(metrics);
            let source = std::sync::Arc::new(
                P2pReadSource::new(endpoint.clone(), key.clone())
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to create p2p source: {e}"))?,
            );
            Ok(options.open(source).await?)
        }
        ScanTarget::AmazonS3 { store, path } => {
            let mut options = session.open_options().with_metrics_registry(metrics);
            if let Some(allocator) = allocator {
                options = options.with_allocator(allocator);
            }

            if s3_ktls_enabled() {
                #[cfg(target_os = "linux")]
                {
                    let source = std::sync::Arc::new(
                        KtlsS3ReadSource::new(store.clone(), path.clone()).await?,
                    );
                    return Ok(options.open(source).await?);
                }
                #[cfg(not(target_os = "linux"))]
                {
                    anyhow::bail!("VORTEX_BENCH_S3_KTLS=1 is only supported on Linux");
                }
            }

            let path_str = path.to_string();
            let store_dyn: std::sync::Arc<dyn ObjectStore> = store.clone();
            Ok(options.open_object_store(&store_dyn, &path_str).await?)
        }
        ScanTarget::ObjectStore { store, path } => {
            let path_str = path.to_string();
            let mut options = session.open_options().with_metrics_registry(metrics);
            if let Some(allocator) = allocator {
                options = options.with_allocator(allocator);
            }
            Ok(options.open_object_store(store, &path_str).await?)
        }
    }
}

async fn open_all_targets(
    targets: &[ScanTarget],
    metrics: std::sync::Arc<dyn MetricsRegistry>,
    concurrency: usize,
    allocator: Option<std::sync::Arc<dyn BufferAllocator>>,
) -> Result<Vec<vortex::file::VortexFile>> {
    let mut files = vec![None; targets.len()];
    let results = futures::stream::iter(targets.iter().enumerate())
        .map(|(idx, target)| {
            let metrics = metrics.clone();
            let allocator = allocator.clone();
            async move {
                let file = open_vortex_file_for_target(target, metrics, allocator).await?;
                Ok::<_, anyhow::Error>((idx, file))
            }
        })
        .buffer_unordered(concurrency.max(1))
        .try_collect::<Vec<_>>()
        .await?;

    for (idx, file) in results {
        files[idx] = Some(file);
    }

    files
        .into_iter()
        .map(|file| file.ok_or_else(|| anyhow::anyhow!("file open missing")))
        .collect()
}

fn object_store_from_url(
    url_str: &str,
) -> Result<(ObjectStoreScheme, ResolvedStore, ObjectStorePath)> {
    let url = Url::parse(url_str)?;
    let (scheme, path) = ObjectStoreScheme::parse(&url).map_err(object_store::Error::from)?;
    let store = match scheme {
        ObjectStoreScheme::Local => {
            ResolvedStore::ObjectStore(std::sync::Arc::new(LocalFileSystem::default()))
        }
        ObjectStoreScheme::AmazonS3 => ResolvedStore::AmazonS3(std::sync::Arc::new(
            AmazonS3Builder::from_env().with_url(url_str).build()?,
        )),
        ObjectStoreScheme::Http => ResolvedStore::ObjectStore(std::sync::Arc::new(
            HttpBuilder::new()
                .with_url(&url[..url::Position::BeforePath])
                .build()?,
        )),
        otherwise => anyhow::bail!("unsupported object store scheme: {otherwise:?}"),
    };

    Ok((scheme, store, path))
}

#[derive(Clone, Debug)]
struct RdmaListedObject {
    key: String,
}

fn rdma_endpoint_and_path(url: &Url) -> Result<(String, String)> {
    endpoint_and_path(url, "rdma")
}

fn p2p_endpoint_and_path(url: &Url) -> Result<(String, String)> {
    endpoint_and_path(url, "p2p")
}

fn endpoint_and_path(url: &Url, scheme: &str) -> Result<(String, String)> {
    anyhow::ensure!(url.scheme() == scheme, "URL scheme must be {scheme}://");
    anyhow::ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "{scheme}:// URLs do not support query or fragment"
    );
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("{scheme}:// URL must include a host"))?;
    let port = url.port().unwrap_or(DEFAULT_RDMA_PORT);
    let key = url.path().trim_start_matches('/').to_string();
    anyhow::ensure!(!key.is_empty(), "{scheme}:// URL path must not be empty");
    Ok((format!("{host}:{port}"), key))
}

async fn rdma_list(endpoint: &str, prefix: &str) -> Result<Vec<RdmaListedObject>> {
    let mut stream = TcpStream::connect(endpoint)
        .await
        .map_err(|e| anyhow::anyhow!("rdma connect to {endpoint} failed: {e}"))?;
    stream.write_u8(OP_LIST).await?;
    write_string(&mut stream, prefix).await?;
    read_status(&mut stream).await?;
    let count = stream.read_u32_le().await? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let key = read_string(&mut stream).await?;
        let _size = stream.read_u64_le().await?;
        out.push(RdmaListedObject { key });
    }
    Ok(out)
}

async fn rdma_size(endpoint: &str, key: &str) -> Result<u64> {
    let mut stream = TcpStream::connect(endpoint)
        .await
        .map_err(|e| anyhow::anyhow!("rdma connect to {endpoint} failed: {e}"))?;
    stream.write_u8(OP_SIZE).await?;
    write_string(&mut stream, key).await?;
    read_status(&mut stream).await?;
    Ok(stream.read_u64_le().await?)
}

async fn p2p_ipc_info(endpoint: &str, key: &str) -> Result<(u64, [u8; 64])> {
    let mut stream = TcpStream::connect(endpoint)
        .await
        .map_err(|e| anyhow::anyhow!("p2p connect to {endpoint} failed: {e}"))?;
    stream.write_u8(OP_IPC_HANDLE).await?;
    write_string(&mut stream, key).await?;
    read_status(&mut stream).await?;
    let size = stream.read_u64_le().await?;
    let mut handle = [0u8; 64];
    stream.read_exact(&mut handle).await?;
    Ok((size, handle))
}

async fn read_all_segments(file: &vortex::file::VortexFile, concurrency: usize) -> Result<()> {
    let segment_count = file.footer().segment_map().len();
    let segment_source = file.segment_source();

    // Pre-register ALL segment requests before polling any of them.
    // request() eagerly sends a ReadEvent::Request to the IO stream,
    // giving the coalescer the full segment picture for optimal merging.
    let futures: Vec<_> = (0..segment_count)
        .map(|idx| {
            let segment_id = vortex::layout::segments::SegmentId::try_from(idx)
                .map_err(|_| anyhow::anyhow!("segment index exceeds u32: {idx}"));
            match segment_id {
                Ok(id) => Ok(segment_source.request(id)),
                Err(e) => Err(e),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    futures::stream::iter(futures)
        .buffer_unordered(concurrency.max(1))
        .try_for_each(|_buffer| async { Ok(()) })
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}

fn io_no_copy_enabled() -> bool {
    std::env::var("VORTEX_BENCH_IO_NO_COPY")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .is_some_and(|v| v != 0)
}

fn s3_ktls_enabled() -> bool {
    std::env::var("VORTEX_BENCH_S3_KTLS")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .is_some_and(|v| v != 0)
}

fn read_env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse::<u64>().ok()
}

fn read_env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse::<usize>().ok()
}

fn read_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .map(|value| value != 0)
        .unwrap_or(default)
}

static P2P_STREAM_REGISTRY: OnceLock<Mutex<Vec<std::sync::Arc<cudarc::driver::CudaStream>>>> =
    OnceLock::new();

fn register_p2p_stream(stream: &std::sync::Arc<cudarc::driver::CudaStream>) {
    let registry = P2P_STREAM_REGISTRY.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = registry.lock();
    if !guard
        .iter()
        .any(|existing| std::sync::Arc::ptr_eq(existing, stream))
    {
        guard.push(stream.clone());
    }
}

fn synchronize_registered_p2p_streams() -> Result<f64> {
    let Some(registry) = P2P_STREAM_REGISTRY.get() else {
        return Ok(0.0);
    };
    let streams = registry.lock().clone();
    if streams.is_empty() {
        return Ok(0.0);
    }
    let start = Instant::now();
    for stream in streams {
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("failed to synchronize p2p stream: {e}"))?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn no_copy_coalesce_distance() -> u64 {
    read_env_u64("VORTEX_BENCH_IO_NO_COPY_COALESCE_DISTANCE").unwrap_or(1024 * 1024)
}

fn no_copy_coalesce_max_size() -> u64 {
    read_env_u64("VORTEX_BENCH_IO_NO_COPY_COALESCE_MAX_SIZE").unwrap_or(16 * 1024 * 1024)
}

#[allow(clippy::cast_possible_truncation)]
fn coalesce_ranges(
    mut ranges: Vec<(u64, usize)>,
    distance: u64,
    max_size: u64,
) -> Vec<(u64, usize)> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_unstable_by_key(|(offset, _)| *offset);

    let mut out = Vec::with_capacity(ranges.len());
    let (mut current_start, first_len) = ranges[0];
    let mut current_end = current_start + first_len as u64;

    for (offset, len) in ranges.into_iter().skip(1) {
        let end = offset + len as u64;
        let gap = offset.saturating_sub(current_end);
        let candidate_end = current_end.max(end);
        let candidate_size = candidate_end - current_start;

        if gap <= distance && candidate_size <= max_size {
            current_end = candidate_end;
        } else {
            out.push((current_start, (current_end - current_start) as usize));
            current_start = offset;
            current_end = end;
        }
    }
    out.push((current_start, (current_end - current_start) as usize));
    out
}

async fn read_all_segments_object_store_no_copy(
    file: &vortex::file::VortexFile,
    store: &std::sync::Arc<dyn ObjectStore>,
    path: &ObjectStorePath,
    concurrency: usize,
) -> Result<u64> {
    let segment_ranges: Vec<(u64, usize)> = file
        .footer()
        .segment_map()
        .iter()
        .map(|segment| (segment.offset, segment.length as usize))
        .collect();
    let ranges = coalesce_ranges(
        segment_ranges,
        no_copy_coalesce_distance(),
        no_copy_coalesce_max_size(),
    );
    let expected: u64 = ranges.iter().map(|(_, len)| *len as u64).sum();
    let total_read = std::sync::Arc::new(AtomicU64::new(0));

    futures::stream::iter(ranges)
        .map(|(offset, len)| {
            let store = store.clone();
            let path = path.clone();
            let total_read = total_read.clone();
            async move {
                let result = store
                    .get_opts(
                        &path,
                        GetOptions {
                            range: Some(GetRange::Bounded(offset..(offset + len as u64))),
                            ..Default::default()
                        },
                    )
                    .await?;

                let mut stream = result.into_stream();
                let mut read = 0usize;
                while let Some(chunk) = stream.next().await {
                    read += chunk?.len();
                }
                anyhow::ensure!(
                    read == len,
                    "Object store stream returned {} bytes but expected {} bytes for range {}..{}",
                    read,
                    len,
                    offset,
                    offset + len as u64
                );
                total_read.fetch_add(read as u64, Ordering::Relaxed);
                Ok::<_, anyhow::Error>(())
            }
        })
        .buffer_unordered(concurrency.max(1))
        .try_collect::<Vec<_>>()
        .await?;

    let read = total_read.load(Ordering::Relaxed);
    anyhow::ensure!(
        read == expected,
        "Object store no-copy read {} bytes but expected {} bytes",
        read,
        expected
    );
    Ok(read)
}

const DEFAULT_RDMA_COALESCING_CONFIG: CoalesceConfig = CoalesceConfig {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};
const DEFAULT_RDMA_CONCURRENCY: usize = 192;
const RDMA_READ_CONCURRENCY_ENV: &str = "VORTEX_RDMA_READ_CONCURRENCY";
const RDMA_COALESCE_DISTANCE_ENV: &str = "VORTEX_RDMA_COALESCE_DISTANCE";
const RDMA_COALESCE_MAX_SIZE_ENV: &str = "VORTEX_RDMA_COALESCE_MAX_SIZE";
const RDMA_COALESCE_DISABLE_ENV: &str = "VORTEX_RDMA_COALESCE_DISABLE";
const RDMA_POOL_SIZE_ENV: &str = "VORTEX_BENCH_RDMA_POOL_SIZE";
const RDMA_PREWARM_ENV: &str = "VORTEX_BENCH_RDMA_PREWARM";

struct RdmaConnection {
    stream: TcpStream,
}

#[derive(Clone)]
struct RdmaReadSource {
    endpoint: std::sync::Arc<str>,
    key: std::sync::Arc<str>,
    uri: std::sync::Arc<str>,
    size: u64,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    max_connections: usize,
    live_connections: std::sync::Arc<AtomicUsize>,
    idle_connections: std::sync::Arc<TokioMutex<Vec<RdmaConnection>>>,
    connection_available: std::sync::Arc<Notify>,
}

impl RdmaReadSource {
    async fn new(endpoint: std::sync::Arc<str>, key: std::sync::Arc<str>) -> Result<Self> {
        let size = rdma_size(endpoint.as_ref(), key.as_ref()).await?;

        let mut coalesce_config = Some(DEFAULT_RDMA_COALESCING_CONFIG);
        if read_env_bool(RDMA_COALESCE_DISABLE_ENV, false) {
            coalesce_config = None;
        } else if let Some(defaults) = coalesce_config.as_mut() {
            if let Some(distance) = read_env_u64(RDMA_COALESCE_DISTANCE_ENV) {
                defaults.distance = distance;
            }
            if let Some(max_size) = read_env_u64(RDMA_COALESCE_MAX_SIZE_ENV) {
                defaults.max_size = max_size;
            }
        }

        let concurrency =
            read_env_usize(RDMA_READ_CONCURRENCY_ENV).unwrap_or(DEFAULT_RDMA_CONCURRENCY);
        let max_connections = read_env_usize(RDMA_POOL_SIZE_ENV)
            .unwrap_or(concurrency)
            .max(1);

        let source = Self {
            uri: std::sync::Arc::from(format!("rdma://{endpoint}/{key}")),
            endpoint,
            key,
            size,
            concurrency,
            coalesce_config,
            max_connections,
            live_connections: std::sync::Arc::new(AtomicUsize::new(0)),
            idle_connections: std::sync::Arc::new(TokioMutex::new(Vec::new())),
            connection_available: std::sync::Arc::new(Notify::new()),
        };

        if read_env_bool(RDMA_PREWARM_ENV, false) {
            source.prewarm_connections().await?;
        }
        Ok(source)
    }

    async fn open_connection(&self) -> Result<RdmaConnection> {
        let stream = TcpStream::connect(self.endpoint.as_ref()).await?;
        stream.set_nodelay(true)?;
        Ok(RdmaConnection { stream })
    }

    async fn prewarm_connections(&self) -> Result<()> {
        for _ in 0..self.max_connections {
            let conn = self.open_connection().await?;
            self.live_connections.fetch_add(1, Ordering::AcqRel);
            self.idle_connections.lock().await.push(conn);
        }
        self.connection_available.notify_waiters();
        Ok(())
    }

    async fn acquire_connection(&self) -> Result<RdmaConnection> {
        loop {
            if let Some(conn) = self.idle_connections.lock().await.pop() {
                return Ok(conn);
            }

            let current = self.live_connections.load(Ordering::Acquire);
            if current < self.max_connections
                && self
                    .live_connections
                    .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                match self.open_connection().await {
                    Ok(conn) => return Ok(conn),
                    Err(err) => {
                        self.live_connections.fetch_sub(1, Ordering::AcqRel);
                        self.connection_available.notify_one();
                        return Err(err);
                    }
                }
            }

            self.connection_available.notified().await;
        }
    }

    async fn release_connection(&self, connection: RdmaConnection) {
        self.idle_connections.lock().await.push(connection);
        self.connection_available.notify_one();
    }

    fn discard_connection(&self) {
        let live = self.live_connections.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(live > 0, "live_connections underflow");
        self.connection_available.notify_one();
    }

    async fn read_range_over_connection(
        &self,
        connection: &mut RdmaConnection,
        offset: u64,
        target: &mut [u8],
    ) -> Result<()> {
        let req_len = u32::try_from(target.len())
            .map_err(|_| anyhow::anyhow!("request length {} exceeds u32", target.len()))?;

        connection.stream.write_u8(OP_READ).await?;
        write_string(&mut connection.stream, self.key.as_ref()).await?;
        connection.stream.write_u64_le(offset).await?;
        connection.stream.write_u32_le(req_len).await?;

        read_status(&mut connection.stream).await?;
        let response_len = connection.stream.read_u32_le().await? as usize;
        anyhow::ensure!(
            response_len == target.len(),
            "rdma response length mismatch: expected {}, got {}",
            target.len(),
            response_len
        );
        connection.stream.read_exact(target).await?;
        Ok(())
    }

    async fn read_range_into(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        if target.is_empty() {
            return Ok(());
        }

        let mut connection = self.acquire_connection().await?;
        let request_result = self
            .read_range_over_connection(&mut connection, offset, target)
            .await;
        match request_result {
            Ok(()) => {
                self.release_connection(connection).await;
                Ok(())
            }
            Err(_) => {
                // Retry once with a fresh connection in case the peer closed idle TCP.
                self.discard_connection();
                let mut fresh = self.acquire_connection().await?;
                match self
                    .read_range_over_connection(&mut fresh, offset, target)
                    .await
                {
                    Ok(()) => {
                        self.release_connection(fresh).await;
                        Ok(())
                    }
                    Err(err) => {
                        self.discard_connection();
                        Err(err)
                    }
                }
            }
        }
    }
}

impl VortexReadAt for RdmaReadSource {
    fn uri(&self) -> Option<&std::sync::Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.size;
        Box::pin(async move { Ok(size) })
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
        // SAFETY: we fill the entire target before returning.
        unsafe { buffer.set_len(length) };
        let target: Box<dyn WriteTarget> = Box::new(buffer);
        self.read_at_into(offset, target)
    }

    fn read_at_into(
        &self,
        offset: u64,
        target: Box<dyn WriteTarget>,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let source = self.clone();
        Box::pin(async move {
            let mut target = target;
            source
                .read_range_into(offset, target.as_mut_slice())
                .await
                .map_err(|e| vortex_err!("rdma read failed at offset {offset}: {e}"))?;
            target.into_handle()
        })
    }
}

const DEFAULT_P2P_COALESCING_CONFIG: CoalesceConfig = CoalesceConfig {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};
const DEFAULT_P2P_CONCURRENCY: usize = 192;
const P2P_READ_CONCURRENCY_ENV: &str = "VORTEX_P2P_READ_CONCURRENCY";
const P2P_COALESCE_DISTANCE_ENV: &str = "VORTEX_P2P_COALESCE_DISTANCE";
const P2P_COALESCE_MAX_SIZE_ENV: &str = "VORTEX_P2P_COALESCE_MAX_SIZE";
const P2P_COALESCE_DISABLE_ENV: &str = "VORTEX_P2P_COALESCE_DISABLE";
const P2P_STREAMS_ENV: &str = "VORTEX_P2P_STREAMS";

struct P2pMappedRegion {
    context: std::sync::Arc<cudarc::driver::CudaContext>,
    base_ptr: sys::CUdeviceptr,
}

impl Drop for P2pMappedRegion {
    fn drop(&mut self) {
        if self.base_ptr == 0 {
            return;
        }
        if self.context.bind_to_thread().is_err() {
            return;
        }
        unsafe {
            let _ = sys::cuIpcCloseMemHandle(self.base_ptr).result();
        }
    }
}

#[derive(Clone)]
struct P2pReadSource {
    uri: std::sync::Arc<str>,
    size: u64,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    streams: std::sync::Arc<Vec<std::sync::Arc<cudarc::driver::CudaStream>>>,
    next_stream: std::sync::Arc<AtomicUsize>,
    mapped: std::sync::Arc<P2pMappedRegion>,
}

impl P2pReadSource {
    async fn new(endpoint: std::sync::Arc<str>, key: std::sync::Arc<str>) -> Result<Self> {
        let (size, handle_bytes) = p2p_ipc_info(endpoint.as_ref(), key.as_ref()).await?;
        let cuda_session = SESSION.cuda_session();
        let context = cuda_session.context().clone();
        context
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("failed to bind CUDA context: {e}"))?;
        let base_ptr = open_ipc_mem_handle(&context, handle_bytes)?;

        let mut coalesce_config = Some(DEFAULT_P2P_COALESCING_CONFIG);
        if read_env_bool(P2P_COALESCE_DISABLE_ENV, false) {
            coalesce_config = None;
        } else if let Some(defaults) = coalesce_config.as_mut() {
            if let Some(distance) = read_env_u64(P2P_COALESCE_DISTANCE_ENV) {
                defaults.distance = distance;
            }
            if let Some(max_size) = read_env_u64(P2P_COALESCE_MAX_SIZE_ENV) {
                defaults.max_size = max_size;
            }
        }

        let concurrency =
            read_env_usize(P2P_READ_CONCURRENCY_ENV).unwrap_or(DEFAULT_P2P_CONCURRENCY);
        let stream_count = read_env_usize(P2P_STREAMS_ENV).unwrap_or(4).max(1);
        let mut streams = Vec::with_capacity(stream_count);
        for _ in 0..stream_count {
            streams.push(
                context
                    .new_stream()
                    .map_err(|e| anyhow::anyhow!("failed to create p2p CUDA stream: {e}"))?,
            );
        }
        for stream in &streams {
            register_p2p_stream(stream);
        }

        Ok(Self {
            uri: std::sync::Arc::from(format!("p2p://{endpoint}/{key}")),
            size,
            concurrency,
            coalesce_config,
            streams: std::sync::Arc::new(streams),
            next_stream: std::sync::Arc::new(AtomicUsize::new(0)),
            mapped: std::sync::Arc::new(P2pMappedRegion { context, base_ptr }),
        })
    }

    fn next_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        let idx = self.next_stream.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        &self.streams[idx]
    }

    fn read_at_device(&self, offset: u64, length: usize) -> VortexResult<BufferHandle> {
        let end = offset.saturating_add(length as u64);
        if end > self.size {
            return Err(vortex_err!(
                "p2p range {}..{} out of bounds for object size {}",
                offset,
                end,
                self.size
            ));
        }

        let stream = self.next_stream();
        let device = unsafe { stream.alloc::<u8>(length) }
            .map_err(|e| vortex_err!("failed to allocate device buffer for p2p read: {e}"))?;
        let (dst_ptr, _) = device.device_ptr(device.stream());
        let src_ptr = self.mapped.base_ptr + offset;
        unsafe {
            result::memcpy_dtod_async(dst_ptr, src_ptr, length, stream.cu_stream())
                .map_err(|e| vortex_err!("p2p d2d copy failed at offset {offset}: {e}"))?;
        }
        Ok(BufferHandle::new_device(std::sync::Arc::new(
            CudaDeviceBuffer::new(device),
        )))
    }
}

fn open_ipc_mem_handle(
    context: &std::sync::Arc<cudarc::driver::CudaContext>,
    handle_bytes: [u8; 64],
) -> Result<sys::CUdeviceptr> {
    context
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("failed to bind CUDA context for IPC open: {e}"))?;
    let handle = bytes_to_ipc_handle(handle_bytes);
    let mut ptr: sys::CUdeviceptr = 0;
    unsafe {
        sys::cuIpcOpenMemHandle_v2(
            &raw mut ptr,
            handle,
            sys::CUipcMem_flags_enum::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
        )
        .result()
        .map_err(|e| anyhow::anyhow!("cuIpcOpenMemHandle_v2 failed: {e}"))?;
    }
    Ok(ptr)
}

fn bytes_to_ipc_handle(bytes: [u8; 64]) -> sys::CUipcMemHandle {
    let mut reserved: [std::ffi::c_char; 64] = [0; 64];
    for (idx, value) in bytes.into_iter().enumerate() {
        reserved[idx] = value as std::ffi::c_char;
    }
    sys::CUipcMemHandle { reserved }
}

impl VortexReadAt for P2pReadSource {
    fn uri(&self) -> Option<&std::sync::Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.size;
        Box::pin(async move { Ok(size) })
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let source = self.clone();
        Box::pin(async move { source.read_at_device(offset, length) })
    }

    fn read_at_into(
        &self,
        offset: u64,
        target: Box<dyn WriteTarget>,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let source = self.clone();
        let length = target.len();
        Box::pin(async move { source.read_at_device(offset, length) })
    }
}

#[cfg(target_os = "linux")]
const DEFAULT_KTLS_COALESCING_CONFIG: CoalesceConfig = CoalesceConfig {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};
#[cfg(target_os = "linux")]
const DEFAULT_KTLS_CONCURRENCY: usize = 192;
#[cfg(target_os = "linux")]
const KTLS_READ_CONCURRENCY_ENV: &str = "VORTEX_S3_READ_CONCURRENCY";
#[cfg(target_os = "linux")]
const KTLS_COALESCE_DISTANCE_ENV: &str = "VORTEX_S3_COALESCE_DISTANCE";
#[cfg(target_os = "linux")]
const KTLS_COALESCE_MAX_SIZE_ENV: &str = "VORTEX_S3_COALESCE_MAX_SIZE";
#[cfg(target_os = "linux")]
const KTLS_COALESCE_DISABLE_ENV: &str = "VORTEX_S3_COALESCE_DISABLE";
#[cfg(target_os = "linux")]
const KTLS_POOL_SIZE_ENV: &str = "VORTEX_BENCH_S3_KTLS_POOL_SIZE";
#[cfg(target_os = "linux")]
const KTLS_PREWARM_ENV: &str = "VORTEX_BENCH_S3_KTLS_PREWARM";

#[cfg(target_os = "linux")]
enum S3Stream {
    Ktls(ktls::KtlsStream<TcpStream>),
    Plain(TcpStream),
}

#[cfg(target_os = "linux")]
impl S3Stream {
    async fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            S3Stream::Ktls(s) => AsyncWriteExt::write_all(s, buf).await,
            S3Stream::Plain(s) => AsyncWriteExt::write_all(s, buf).await,
        }
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        match self {
            S3Stream::Ktls(s) => AsyncWriteExt::flush(s).await,
            S3Stream::Plain(s) => AsyncWriteExt::flush(s).await,
        }
    }

    async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            S3Stream::Ktls(s) => AsyncReadExt::read(s, buf).await,
            S3Stream::Plain(s) => AsyncReadExt::read(s, buf).await,
        }
    }
}

#[cfg(target_os = "linux")]
struct KtlsConnection {
    stream: S3Stream,
}

#[cfg(target_os = "linux")]
#[derive(Clone)]
struct KtlsS3ReadSource {
    store: std::sync::Arc<AmazonS3>,
    path: ObjectStorePath,
    uri: std::sync::Arc<str>,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    tls: Option<std::sync::Arc<TlsConnector>>,
    host: std::sync::Arc<str>,
    host_header: std::sync::Arc<str>,
    port: u16,
    request_target: std::sync::Arc<str>,
    use_tls: bool,
    max_connections: usize,
    live_connections: std::sync::Arc<AtomicUsize>,
    idle_connections: std::sync::Arc<TokioMutex<Vec<KtlsConnection>>>,
    connection_available: std::sync::Arc<Notify>,
}

#[cfg(target_os = "linux")]
impl KtlsS3ReadSource {
    async fn new(store: std::sync::Arc<AmazonS3>, path: ObjectStorePath) -> Result<Self> {
        let signed = store
            .signed_url(
                reqwest::Method::GET,
                &path,
                Duration::from_secs(6 * 60 * 60),
            )
            .await?;
        let host = signed
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("signed S3 URL is missing host"))?
            .to_string();
        let port = signed
            .port_or_known_default()
            .ok_or_else(|| anyhow::anyhow!("could not determine S3 port from signed URL"))?;
        let request_target = match signed.query() {
            Some(query) => format!("{}?{query}", signed.path()),
            None => signed.path().to_string(),
        };
        let use_tls = signed.scheme() == "https";
        let host_header = if (port == 443 && use_tls) || (port == 80 && !use_tls) {
            host.clone()
        } else {
            format!("{host}:{port}")
        };

        let tls = if use_tls {
            let mut roots = RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let mut tls_config = ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth();
            tls_config.enable_secret_extraction = true;
            Some(TlsConnector::from(std::sync::Arc::new(tls_config)))
        } else {
            None
        };

        let mut coalesce_config = Some(DEFAULT_KTLS_COALESCING_CONFIG);
        if read_env_bool(KTLS_COALESCE_DISABLE_ENV, false) {
            coalesce_config = None;
        } else if let Some(defaults) = coalesce_config.as_mut() {
            if let Some(distance) = read_env_u64(KTLS_COALESCE_DISTANCE_ENV) {
                defaults.distance = distance;
            }
            if let Some(max_size) = read_env_u64(KTLS_COALESCE_MAX_SIZE_ENV) {
                defaults.max_size = max_size;
            }
        }

        let concurrency =
            read_env_usize(KTLS_READ_CONCURRENCY_ENV).unwrap_or(DEFAULT_KTLS_CONCURRENCY);
        let max_connections = read_env_usize(KTLS_POOL_SIZE_ENV)
            .unwrap_or(concurrency)
            .max(1);

        let source = Self {
            store,
            path: path.clone(),
            uri: std::sync::Arc::from(path.to_string()),
            concurrency,
            coalesce_config,
            tls: tls.map(std::sync::Arc::new),
            host: std::sync::Arc::from(host),
            host_header: std::sync::Arc::from(host_header),
            port,
            request_target: std::sync::Arc::from(request_target),
            use_tls,
            max_connections,
            live_connections: std::sync::Arc::new(AtomicUsize::new(0)),
            idle_connections: std::sync::Arc::new(TokioMutex::new(Vec::new())),
            connection_available: std::sync::Arc::new(Notify::new()),
        };

        if read_env_bool(KTLS_PREWARM_ENV, false) {
            source.prewarm_connections().await?;
        }

        Ok(source)
    }

    async fn open_connection(&self) -> Result<KtlsConnection> {
        let addr = format!("{}:{}", self.host, self.port);
        let tcp = TcpStream::connect(addr).await?;
        tcp.set_nodelay(true)?;

        let stream = if let Some(tls) = &self.tls {
            let server_name = ServerName::try_from(self.host.as_ref().to_string())
                .map_err(|e| anyhow::anyhow!("invalid TLS server name '{}': {e}", self.host))?;
            let tls_stream = tls.connect(server_name, CorkStream::new(tcp)).await?;
            let ktls_stream = ktls::config_ktls_client(tls_stream).await?;
            S3Stream::Ktls(ktls_stream)
        } else {
            S3Stream::Plain(tcp)
        };
        Ok(KtlsConnection { stream })
    }

    async fn prewarm_connections(&self) -> Result<()> {
        for _ in 0..self.max_connections {
            let connection = self.open_connection().await?;
            self.live_connections.fetch_add(1, Ordering::AcqRel);
            self.idle_connections.lock().await.push(connection);
        }
        self.connection_available.notify_waiters();
        Ok(())
    }

    async fn acquire_connection(&self) -> Result<KtlsConnection> {
        loop {
            if let Some(connection) = self.idle_connections.lock().await.pop() {
                return Ok(connection);
            }

            let current = self.live_connections.load(Ordering::Acquire);
            if current < self.max_connections
                && self
                    .live_connections
                    .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                match self.open_connection().await {
                    Ok(connection) => return Ok(connection),
                    Err(err) => {
                        self.live_connections.fetch_sub(1, Ordering::AcqRel);
                        self.connection_available.notify_one();
                        return Err(err);
                    }
                }
            }

            self.connection_available.notified().await;
        }
    }

    async fn release_connection(&self, connection: KtlsConnection) {
        self.idle_connections.lock().await.push(connection);
        self.connection_available.notify_one();
    }

    fn discard_connection(&self) {
        let live = self.live_connections.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(live > 0, "live_connections underflow");
        self.connection_available.notify_one();
    }

    async fn read_range_over_connection(
        &self,
        connection: &mut KtlsConnection,
        offset: u64,
        target: &mut [u8],
    ) -> Result<bool> {
        let end = offset + (target.len() as u64) - 1;
        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nRange: bytes={offset}-{end}\r\nAccept: */*\r\nConnection: keep-alive\r\n\r\n",
            self.request_target, self.host_header
        );
        connection.stream.write_all(request.as_bytes()).await?;
        connection.stream.flush().await?;

        let mut response_buf = Vec::with_capacity(16 * 1024);
        let mut scratch = [0u8; 8192];
        let header_end = loop {
            let read = connection.stream.read(&mut scratch).await?;
            if read == 0 {
                anyhow::bail!("unexpected EOF while reading S3 HTTP response headers");
            }
            response_buf.extend_from_slice(&scratch[..read]);
            if let Some(pos) = find_header_end(&response_buf) {
                break pos + 4;
            }
            if response_buf.len() > 256 * 1024 {
                anyhow::bail!("S3 HTTP headers exceeded 256KB");
            }
        };

        let headers = &response_buf[..header_end];
        let status = parse_status_code(headers)?;
        if status != 206 {
            anyhow::bail!("expected HTTP 206 for ranged S3 read, got {status}");
        }
        if header_has_chunked_encoding(headers) {
            anyhow::bail!("chunked S3 responses are not supported in kTLS path");
        }
        let content_length = parse_content_length(headers)?;
        if content_length != target.len() {
            anyhow::bail!(
                "S3 content-length mismatch: expected {}, got {}",
                target.len(),
                content_length
            );
        }

        let mut filled = 0usize;
        let prefetched = &response_buf[header_end..];
        if prefetched.len() > target.len() {
            anyhow::bail!(
                "S3 prefetched body exceeded expected length: {} > {}",
                prefetched.len(),
                target.len()
            );
        }
        if !prefetched.is_empty() {
            target[..prefetched.len()].copy_from_slice(prefetched);
            filled += prefetched.len();
        }
        while filled < target.len() {
            let read = connection.stream.read(&mut target[filled..]).await?;
            if read == 0 {
                anyhow::bail!(
                    "unexpected EOF while reading S3 body (read {} of {} bytes)",
                    filled,
                    target.len()
                );
            }
            filled += read;
        }

        Ok(!header_has_connection_close(headers))
    }

    async fn read_range_into(&self, offset: u64, target: &mut [u8]) -> Result<()> {
        if target.is_empty() {
            return Ok(());
        }

        let mut connection = self.acquire_connection().await?;
        let request_result = self
            .read_range_over_connection(&mut connection, offset, target)
            .await;
        match request_result {
            Ok(true) => {
                self.release_connection(connection).await;
                Ok(())
            }
            Ok(false) => {
                self.discard_connection();
                Ok(())
            }
            Err(_) => {
                // Connection may be stale (S3 closed idle keep-alive).
                // Retry once with a fresh connection.
                self.discard_connection();
                let mut fresh = self.acquire_connection().await?;
                match self
                    .read_range_over_connection(&mut fresh, offset, target)
                    .await
                {
                    Ok(true) => {
                        self.release_connection(fresh).await;
                        Ok(())
                    }
                    Ok(false) => {
                        self.discard_connection();
                        Ok(())
                    }
                    Err(err) => {
                        self.discard_connection();
                        Err(err)
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl VortexReadAt for KtlsS3ReadSource {
    fn uri(&self) -> Option<&std::sync::Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = self.store.clone();
        let path = self.path.clone();
        Box::pin(async move {
            Ok(store
                .head(&path)
                .await
                .map_err(|e| vortex_err!("s3 head failed: {e}"))?
                .size)
        })
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
        // SAFETY: we fill the full target before returning.
        unsafe { buffer.set_len(length) };
        let target: Box<dyn WriteTarget> = Box::new(buffer);
        self.read_at_into(offset, target)
    }

    fn read_at_into(
        &self,
        offset: u64,
        target: Box<dyn WriteTarget>,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let source = self.clone();
        Box::pin(async move {
            let mut target = target;
            source
                .read_range_into(offset, target.as_mut_slice())
                .await
                .map_err(|e| vortex_err!("kTLS S3 ranged read failed at offset {offset}: {e}"))?;
            target.into_handle()
        })
    }
}

#[cfg(target_os = "linux")]
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|window| window == b"\r\n\r\n")
}

#[cfg(target_os = "linux")]
fn parse_status_code(headers: &[u8]) -> Result<u16> {
    let header_str = std::str::from_utf8(headers)?;
    let status_line = header_str
        .split("\r\n")
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing HTTP status line"))?;
    let code = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("missing HTTP status code"))?;
    Ok(u16::from_str(code)?)
}

#[cfg(target_os = "linux")]
fn parse_content_length(headers: &[u8]) -> Result<usize> {
    let header_str = std::str::from_utf8(headers)?;
    for line in header_str.split("\r\n").skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            return Ok(value.trim().parse::<usize>()?);
        }
    }
    anyhow::bail!("missing content-length header")
}

#[cfg(target_os = "linux")]
fn header_has_chunked_encoding(headers: &[u8]) -> bool {
    let Ok(header_str) = std::str::from_utf8(headers) else {
        return false;
    };

    for line in header_str.split("\r\n").skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("transfer-encoding")
        {
            return value
                .split(',')
                .any(|encoding| encoding.trim().eq_ignore_ascii_case("chunked"));
        }
    }
    false
}

#[cfg(target_os = "linux")]
fn header_has_connection_close(headers: &[u8]) -> bool {
    let Ok(header_str) = std::str::from_utf8(headers) else {
        return true;
    };

    for line in header_str.split("\r\n").skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("connection")
        {
            return value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("close"));
        }
    }
    false
}

#[derive(Clone)]
struct BenchLayoutReader {
    inner: std::sync::Arc<dyn LayoutReader>,
    segment_pruning: bool,
    bypass_filter: bool,
}

impl BenchLayoutReader {
    fn new(
        inner: std::sync::Arc<dyn LayoutReader>,
        segment_pruning: bool,
        bypass_filter: bool,
    ) -> Self {
        Self {
            inner,
            segment_pruning,
            bypass_filter,
        }
    }
}

impl LayoutReader for BenchLayoutReader {
    fn name(&self) -> &std::sync::Arc<str> {
        self.inner.name()
    }

    fn dtype(&self) -> &vortex::dtype::DType {
        self.inner.dtype()
    }

    fn row_count(&self) -> u64 {
        self.inner.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[vortex::dtype::FieldMask],
        row_range: &std::ops::Range<u64>,
        splits: &mut std::collections::BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.inner.register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &std::ops::Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        if !self.segment_pruning {
            return self.inner.pruning_evaluation(row_range, expr, mask);
        }

        let len = mask.len();
        let fut = self.inner.pruning_evaluation(row_range, expr, mask)?;
        Ok(MaskFuture::new(len, async move {
            let mask = fut.await?;
            if mask.all_false() {
                Ok(mask)
            } else {
                Ok(Mask::new_true(len))
            }
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &std::ops::Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if self.bypass_filter {
            Ok(mask)
        } else {
            self.inner.filter_evaluation(row_range, expr, mask)
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &std::ops::Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<vortex::array::ArrayRef>>> {
        self.inner.projection_evaluation(row_range, expr, mask)
    }
}

fn print_stats(pinned_pool: Option<&PinnedByteBufferPool>) {
    let io = io_request_stats();
    if io.registered > 0 || io.dispatched > 0 || io.completed > 0 {
        println!(
            "io_requests: registered={} polled={} dispatched={} completed={} max_in_flight={}",
            io.registered, io.polled, io.dispatched, io.completed, io.max_in_flight
        );
    }
    let copy = copy_stats();
    if copy.bytes > 0 && copy.nanos > 0 {
        let gb_per_s = copy.bytes as f64 / (copy.nanos as f64 / 1e9) / 1e9;
        let ms = copy.nanos as f64 / 1e6;
        println!(
            "in_memory_memcpy: {:.2} GB/s ({:.2} ms total, {} copies)",
            gb_per_s, ms, copy.count
        );
    }
    let alloc = default_alloc_stats();
    if alloc.count > 0 {
        println!("default_allocs: {} ({} bytes)", alloc.count, alloc.bytes);
    }
    if let Some(pool) = pinned_pool {
        let stats = pool.stats();
        println!(
            "pinned_pool: hits={} misses={} allocs={} puts={}",
            stats.hits, stats.misses, stats.allocs, stats.puts
        );
    }
}
