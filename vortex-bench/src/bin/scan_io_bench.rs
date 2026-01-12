// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use futures::StreamExt;
use futures::TryStreamExt;
use object_store::ObjectStore;
use object_store::ObjectStoreScheme;
use object_store::aws::AmazonS3Builder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectStorePath;
use url::Url;
use vortex::array::Array;
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
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::metrics::VortexMetrics;
use vortex_bench::SESSION;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

#[derive(Parser, Debug)]
#[command(version, about = "Benchmark Vortex scans over local files vs object stores")]
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
    let args = Args::parse();

    let (projection, filter) = build_scan_exprs(&args)?;
    let metrics = VortexMetrics::new_with_tags([("bench", "scan-io")]);
    let read_bytes = metrics.counter("vortex.io.read.total_size");

    let targets = resolve_targets(&args).await?;
    let cached_files = if args.reopen {
        None
    } else {
        Some(std::sync::Arc::new(
            open_all_targets(&targets, metrics.clone()).await?,
        ))
    };
    let mut total_rows = 0usize;
    let mut total_elapsed = 0.0f64;
    let mut total_bytes = 0i64;
    let mut total_first_latency = 0.0f64;
    let mut total_first_bytes = 0i64;

    for _ in 0..args.iterations {
        read_bytes.clear();

        let start = Instant::now();
        let bytes_before = read_bytes.count();
        let first_seen = std::sync::Arc::new(AtomicBool::new(false));
        let first_info = std::sync::Arc::new(Mutex::new(None::<(f64, i64)>));

        let rows = futures::stream::iter(targets.iter().enumerate())
            .map(|(idx, target)| {
                let cached_files = cached_files.clone();
                let projection = projection.clone();
                let filter = filter.clone();
                let metrics = metrics.clone();
                let read_bytes = read_bytes.clone();
                let first_seen = first_seen.clone();
                let first_info = first_info.clone();
                let start = start;
                async move {
                    let file = match &cached_files {
                        Some(files) => files[idx].clone(),
                        None => open_vortex_file_for_target(target, metrics).await?,
                    };
                    let scan = file
                        .scan()?
                        .with_projection(projection)
                        .with_some_filter(filter)
                        .with_concurrency(args.concurrency);

                    let mut stream = scan.into_stream()?;
                    let mut file_rows = 0usize;
                    while let Some(array) = stream.try_next().await? {
                        if !first_seen.load(Ordering::Relaxed)
                            && !first_seen.swap(true, Ordering::Relaxed)
                        {
                            let latency = start.elapsed().as_secs_f64();
                            let bytes = read_bytes.count() - bytes_before;
                            *first_info.lock().expect("first_info lock poisoned") =
                                Some((latency, bytes));
                        }
                        file_rows += array.len();
                    }

                    drop(file);
                    Ok::<_, anyhow::Error>(file_rows)
                }
            })
            .buffer_unordered(args.file_concurrency.max(1))
            .try_fold(0usize, |rows, file_rows| async move { Ok(rows + file_rows) })
            .await?;

        let elapsed = start.elapsed().as_secs_f64();
        let bytes = read_bytes.count();

        total_rows += rows;
        total_elapsed += elapsed;
        total_bytes += bytes;
        let (iter_first_latency, iter_first_bytes) = first_info
            .lock()
            .expect("first_info lock poisoned")
            .unwrap_or((elapsed, read_bytes.count() - bytes_before));
        total_first_latency += iter_first_latency;
        total_first_bytes += iter_first_bytes;

    }

    let avg_elapsed = total_elapsed / args.iterations as f64;
    let avg_bytes = total_bytes as f64 / args.iterations as f64;
    let avg_first_latency = total_first_latency / args.iterations as f64;
    let avg_first_bytes = total_first_bytes as f64 / args.iterations as f64;
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
    println!("rows={}", total_rows / args.iterations);
    println!("avg_time_s={:.3}", avg_elapsed);
    println!("avg_bytes={}", avg_bytes as i64);
    println!("avg_mb_s={:.2}", total_mb_s);
    println!("avg_first_latency_ms={:.2}", avg_first_latency * 1000.0);
    println!("steady_mb_s={:.2}", steady_mb_s);

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
                LiteralType::I16 => lit(value as i16),
                LiteralType::I32 => lit(value as i32),
                LiteralType::I64 => lit(value as i64),
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
    ObjectStore {
        store: std::sync::Arc<dyn ObjectStore>,
        path: ObjectStorePath,
    },
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
            let mut entries = store.list(Some(&path));
            let mut targets = Vec::new();
            while let Some(entry) = entries.try_next().await? {
                targets.push(ScanTarget::ObjectStore {
                    store: store.clone(),
                    path: entry.location.clone(),
                });
            }
            return Ok(targets);
        }

        return Ok(vec![ScanTarget::ObjectStore {
            store,
            path,
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
    metrics: VortexMetrics,
) -> Result<vortex::file::VortexFile> {
    let session = SESSION.clone();
    match target {
        ScanTarget::Local(path) => Ok(session
            .open_options()
            .with_metrics(metrics)
            .open(path.clone())
            .await?),
        ScanTarget::ObjectStore { store, path } => {
            let path_str = path.to_string();
            Ok(session
                .open_options()
                .with_metrics(metrics)
                .open_object_store(store, &path_str)
                .await?)
        }
    }
}

async fn open_all_targets(
    targets: &[ScanTarget],
    metrics: VortexMetrics,
) -> Result<Vec<vortex::file::VortexFile>> {
    let mut files = Vec::with_capacity(targets.len());
    for target in targets {
        files.push(open_vortex_file_for_target(target, metrics.clone()).await?);
    }
    Ok(files)
}

fn object_store_from_url(
    url_str: &str,
) -> Result<(ObjectStoreScheme, std::sync::Arc<dyn ObjectStore>, ObjectStorePath)> {
    let url = Url::parse(url_str)?;
    let (scheme, path) = ObjectStoreScheme::parse(&url).map_err(object_store::Error::from)?;
    let store: std::sync::Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => std::sync::Arc::new(LocalFileSystem::default()),
        ObjectStoreScheme::AmazonS3 => {
            std::sync::Arc::new(AmazonS3Builder::from_env().with_url(url_str).build()?)
        }
        ObjectStoreScheme::Http => std::sync::Arc::new(
            HttpBuilder::new()
                .with_url(&url[..url::Position::BeforePath])
                .build()?,
        ),
        otherwise => anyhow::bail!("unsupported object store scheme: {otherwise:?}"),
    };

    Ok((scheme, store, path))
}
