// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
#[cfg(target_os = "linux")]
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
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
use tracing_subscriber::EnvFilter;
use url::Url;
use vortex::array::Array;
use vortex::array::MaskFuture;
#[cfg(target_os = "linux")]
use vortex::array::buffer::BufferHandle;
use vortex::array::expr::Expression;
use vortex::array::expr::col;
use vortex::array::expr::eq;
use vortex::array::expr::gt;
use vortex::array::expr::gt_eq;
use vortex::array::expr::lit;
use vortex::array::expr::lt;
use vortex::array::expr::lt_eq;
use vortex::array::expr::not_eq;
use vortex::array::expr::root;
use vortex::array::expr::select;
#[cfg(target_os = "linux")]
use vortex::buffer::Alignment;
#[cfg(target_os = "linux")]
use vortex::buffer::ByteBufferMut;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::segments::io_request_stats;
use vortex::file::segments::reset_io_request_stats;
use vortex::io::BufferAllocator;
#[cfg(target_os = "linux")]
use vortex::io::CoalesceConfig;
#[cfg(target_os = "linux")]
use vortex::io::VortexReadAt;
#[cfg(target_os = "linux")]
use vortex::io::WriteTarget;
use vortex::io::copy_stats;
use vortex::io::default_alloc_stats;
use vortex::io::reset_copy_stats;
use vortex::io::reset_default_alloc_stats;
use vortex::layout::LayoutReader;
use vortex::mask::Mask;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::MetricValue;
use vortex::metrics::MetricsRegistry;
use vortex_bench::SESSION;
use vortex_cuda::CudaSessionExt;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PinnedDeviceAllocator;
use vortex_scan::ScanBuilder;

#[cfg(target_os = "linux")]
use ktls::CorkStream;
#[cfg(target_os = "linux")]
use rustls::ClientConfig;
#[cfg(target_os = "linux")]
use rustls::RootCertStore;
#[cfg(target_os = "linux")]
use rustls::pki_types::ServerName;
#[cfg(target_os = "linux")]
use tokio::io::AsyncReadExt;
#[cfg(target_os = "linux")]
use tokio::io::AsyncWriteExt;
#[cfg(target_os = "linux")]
use tokio::net::TcpStream;
#[cfg(target_os = "linux")]
use tokio::sync::Mutex as TokioMutex;
#[cfg(target_os = "linux")]
use tokio::sync::Notify;
#[cfg(target_os = "linux")]
use tokio_rustls::TlsConnector;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Benchmark Vortex scans over local files vs object stores"
)]
struct Args {
    /// File path, directory, or object store URL (e.g. file:/..., s3://bucket/path, https://host/path)
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

    let (projection, filter) = build_scan_exprs(&args)?;
    let metrics: std::sync::Arc<dyn MetricsRegistry> =
        std::sync::Arc::new(DefaultMetricsRegistry::default());
    let total_read_bytes = {
        let metrics = metrics.clone();
        move || -> u64 {
            metrics
                .snapshot()
                .iter()
                .filter(|m| m.name().as_ref() == "vortex.io.read.total_size")
                .filter_map(|m| match m.value() {
                    MetricValue::Counter(c) => Some(c.value()),
                    _ => None,
                })
                .sum()
        }
    };

    #[allow(clippy::if_then_some_else_none)]
    let (gpu_allocator, pinned_pool) = if args.gpu {
        let cuda_session = SESSION.cuda_session();
        vortex_cuda::layout::register_cuda_layout(&SESSION);
        let pool = std::sync::Arc::new(PinnedByteBufferPool::new(cuda_session.context().clone()));
        let allocator =
            std::sync::Arc::new(PinnedDeviceAllocator::from_session(pool.clone(), &SESSION)?);
        (Some(allocator), Some(pool))
    } else {
        (None, None)
    };
    let allocator: Option<std::sync::Arc<dyn BufferAllocator>> = gpu_allocator
        .as_ref()
        .map(|alloc| alloc.clone() as std::sync::Arc<dyn BufferAllocator>);

    let targets = resolve_targets(&args).await?;
    let cached_files = if args.reopen {
        None
    } else {
        Some(std::sync::Arc::new(
            open_all_targets(
                &targets,
                metrics.clone(),
                args.file_concurrency,
                allocator.clone(),
            )
            .await?,
        ))
    };
    reset_default_alloc_stats();
    reset_copy_stats();
    reset_io_request_stats();
    if let Some(pool) = pinned_pool.as_ref() {
        pool.reset_stats();
    }

    let start = Instant::now();
    let bytes_before = total_read_bytes();
    let manual_io_bytes = std::sync::Arc::new(AtomicU64::new(0));
    let first_seen = std::sync::Arc::new(AtomicBool::new(false));
    let first_info = std::sync::Arc::new(Mutex::new(None::<(f64, u64)>));
    let targets = targets.clone();

    let rows = futures::stream::iter(0..args.iterations)
        .flat_map(|_| futures::stream::iter(targets.clone().into_iter().enumerate()))
        .map(|(idx, target)| {
            let cached_files = cached_files.clone();
            let projection = projection.clone();
            let filter = filter.clone();
            let metrics = metrics.clone();
            let total_read_bytes = total_read_bytes.clone();
            let first_seen = first_seen.clone();
            let first_info = first_info.clone();
            let mode = mode.clone();
            let allocator = allocator.clone();
            let manual_io_bytes = manual_io_bytes.clone();
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
                                let io_bytes = read_all_segments_object_store_no_copy(
                                    &file,
                                    store,
                                    path,
                                    args.concurrency,
                                )
                                .await?;
                                manual_io_bytes.fetch_add(io_bytes, Ordering::Relaxed);
                            }
                            ScanTarget::AmazonS3 { store, path } => {
                                let store_dyn: std::sync::Arc<dyn ObjectStore> = store.clone();
                                let io_bytes = read_all_segments_object_store_no_copy(
                                    &file,
                                    &store_dyn,
                                    path,
                                    args.concurrency,
                                )
                                .await?;
                                manual_io_bytes.fetch_add(io_bytes, Ordering::Relaxed);
                            }
                            ScanTarget::Local(_) => {
                                read_all_segments(&file, args.concurrency).await?;
                            }
                        }
                    } else {
                        read_all_segments(&file, args.concurrency).await?;
                    }
                    if !first_seen.load(Ordering::Relaxed)
                        && !first_seen.swap(true, Ordering::Relaxed)
                    {
                        let latency = start.elapsed().as_secs_f64();
                        let bytes = (total_read_bytes() - bytes_before)
                            + manual_io_bytes.load(Ordering::Relaxed);
                        *first_info.lock() = Some((latency, bytes));
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
                        let latency = start.elapsed().as_secs_f64();
                        let bytes = (total_read_bytes() - bytes_before)
                            + manual_io_bytes.load(Ordering::Relaxed);
                        *first_info.lock() = Some((latency, bytes));
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
    let gpu_sync_ms = if let Some(allocator) = gpu_allocator {
        let sync_start = Instant::now();
        allocator.synchronize()?;
        sync_start.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let bytes = total_read_bytes() + manual_io_bytes.load(Ordering::Relaxed);
    let (first_latency, first_bytes) =
        (*first_info.lock()).unwrap_or_else(|| (elapsed, bytes - bytes_before));

    let avg_elapsed = elapsed / args.iterations as f64;
    let avg_bytes = bytes as f64 / args.iterations as f64;
    let avg_first_latency = first_latency / args.iterations as f64;
    let avg_first_bytes = first_bytes as f64 / args.iterations as f64;
    let steady_bytes = (avg_bytes - avg_first_bytes).max(0.0);
    let steady_time = (avg_elapsed - avg_first_latency).max(0.0);
    let total_mb_s = if avg_elapsed > 0.0 {
        avg_bytes / (1024.0 * 1024.0) / avg_elapsed
    } else {
        0.0
    };
    let steady_mb_s = if steady_time > 0.0 {
        steady_bytes / (1024.0 * 1024.0) / steady_time
    } else {
        0.0
    };

    println!("files={}", targets.len());
    println!("rows={}", rows / args.iterations);
    println!("avg_time_s={:.3}", avg_elapsed);
    println!("avg_bytes={:.0}", avg_bytes);
    println!("avg_mb_s={:.2}", total_mb_s);
    println!("avg_first_latency_ms={:.2}", avg_first_latency * 1000.0);
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
    if args.gpu {
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

async fn read_all_segments(file: &vortex::file::VortexFile, concurrency: usize) -> Result<()> {
    let segment_count = file.footer().segment_map().len();
    let segment_source = file.segment_source();

    futures::stream::iter(0..segment_count)
        .map(|idx| {
            let segment_source = segment_source.clone();
            async move {
                let segment_id = vortex::layout::segments::SegmentId::try_from(idx)
                    .map_err(|_| anyhow::anyhow!("segment index exceeds u32: {idx}"))?;
                let buffer = segment_source.request(segment_id).await?;
                drop(buffer);
                Ok::<_, anyhow::Error>(())
            }
        })
        .buffer_unordered(concurrency.max(1))
        .try_collect::<Vec<_>>()
        .await?;

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

#[cfg(target_os = "linux")]
fn read_env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse::<usize>().ok()
}

#[cfg(target_os = "linux")]
fn read_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .map(|value| value != 0)
        .unwrap_or(default)
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
struct KtlsConnection {
    stream: ktls::KtlsStream<TcpStream>,
}

#[cfg(target_os = "linux")]
#[derive(Clone)]
struct KtlsS3ReadSource {
    store: std::sync::Arc<AmazonS3>,
    path: ObjectStorePath,
    uri: std::sync::Arc<str>,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    tls: std::sync::Arc<TlsConnector>,
    host: std::sync::Arc<str>,
    host_header: std::sync::Arc<str>,
    port: u16,
    request_target: std::sync::Arc<str>,
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
        let host_header = if port == 443 {
            host.clone()
        } else {
            format!("{host}:{port}")
        };

        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let mut tls_config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        tls_config.enable_secret_extraction = true;
        let tls = TlsConnector::from(std::sync::Arc::new(tls_config));

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
            tls: std::sync::Arc::new(tls),
            host: std::sync::Arc::from(host),
            host_header: std::sync::Arc::from(host_header),
            port,
            request_target: std::sync::Arc::from(request_target),
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

        let server_name = ServerName::try_from(self.host.as_ref().to_string())
            .map_err(|e| anyhow::anyhow!("invalid TLS server name '{}': {e}", self.host))?;
        let tls_stream = self.tls.connect(server_name, CorkStream::new(tcp)).await?;
        let stream = ktls::config_ktls_client(tls_stream).await?;
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
