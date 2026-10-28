// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Collect request-level S3 measurements without a query engine or a latency model.

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use clap::Parser;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use mimalloc::MiMalloc;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use serde_json::json;
use url::Url;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
struct Args {
    /// Read-only dataset prefix. Objects must fit the largest requested range.
    #[arg(long)]
    source: Url,
    #[arg(long)]
    output: PathBuf,
    /// Identity of this measurement window, retained separately from the actual CI run ID.
    #[arg(long)]
    measurement_id: String,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "4096,65536,262144,1048576,8388608"
    )]
    sizes: Vec<u64>,
    #[arg(long, value_delimiter = ',', default_value = "1,4,16,64")]
    concurrency: Vec<usize>,
    #[arg(long, default_value_t = 64)]
    requests: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 20261028)]
    seed: u64,
    #[arg(long, default_value_t = 20.0)]
    warmup_seconds: f64,
    #[arg(long, default_value_t = 60.0)]
    timeout_seconds: f64,
    /// Exercise the collector using files; these traces are ineligible for S3 model training.
    #[arg(long)]
    allow_local: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Request {
    kind: &'static str,
    phase: &'static str,
    operation: &'static str,
    round: usize,
    ordinal: usize,
    path: String,
    range: Option<Range<u64>>,
    object_size: Option<u64>,
    e_tag: Option<String>,
    version: Option<String>,
    requested_concurrency: usize,
    inflight_at_start: usize,
    start_ms: f64,
    started_unix_ms: u128,
    response_ms: Option<f64>,
    first_byte_ms: Option<f64>,
    duration_ms: f64,
    bytes: u64,
    status: &'static str,
    error: Option<String>,
}

impl Request {
    fn new(operation: &'static str, path: &Path, epoch: &Epoch) -> Self {
        let elapsed = epoch.start.elapsed();
        let start_ms = elapsed.as_secs_f64() * 1000.0;
        Self {
            kind: "request",
            phase: "catalog",
            operation,
            round: 0,
            ordinal: 0,
            path: path.to_string(),
            range: None,
            object_size: None,
            e_tag: None,
            version: None,
            requested_concurrency: 1,
            inflight_at_start: 1,
            start_ms,
            started_unix_ms: epoch.unix_ms + elapsed.as_millis(),
            response_ms: None,
            first_byte_ms: None,
            duration_ms: 0.0,
            bytes: 0,
            status: "complete",
            error: None,
        }
    }
}

struct Epoch {
    start: Instant,
    unix_ms: u128,
}

#[derive(Default)]
struct Inflight {
    current: AtomicUsize,
    peak: AtomicUsize,
}

struct Active<'a>(&'a Inflight);

impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.current.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Read<'a> {
    object: &'a ObjectMeta,
    range: Range<u64>,
    phase: &'static str,
    round: usize,
    ordinal: usize,
    concurrency: usize,
}

async fn measure_get(
    store: &dyn ObjectStore,
    read: Read<'_>,
    epoch: &Epoch,
    inflight: &Inflight,
    timeout: Duration,
) -> Request {
    let mut record = Request::new("GET", &read.object.location, epoch);
    record.phase = read.phase;
    record.round = read.round;
    record.ordinal = read.ordinal;
    record.range = Some(read.range.clone());
    record.object_size = Some(read.object.size);
    record.e_tag = read.object.e_tag.clone();
    record.version = read.object.version.clone();
    record.requested_concurrency = read.concurrency;
    record.inflight_at_start = inflight.current.fetch_add(1, Ordering::Relaxed) + 1;
    inflight
        .peak
        .fetch_max(record.inflight_at_start, Ordering::Relaxed);
    let _active = Active(inflight);
    let start = Instant::now();
    let result = tokio::time::timeout(timeout, async {
        let response = store
            .get_opts(
                &read.object.location,
                GetOptions {
                    range: Some(GetRange::Bounded(read.range.clone())),
                    if_match: read.object.e_tag.clone(),
                    version: read.object.version.clone(),
                    ..GetOptions::default()
                },
            )
            .await?;
        record.response_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
        anyhow::ensure!(
            response.range == read.range,
            "S3 returned a different range"
        );
        anyhow::ensure!(
            response.meta.size == read.object.size && response.meta.e_tag == read.object.e_tag,
            "Object identity changed during measurement"
        );
        record.version = response.meta.version.clone();
        let mut body = response.into_stream();
        while let Some(chunk) = body.try_next().await? {
            if !chunk.is_empty() {
                record
                    .first_byte_ms
                    .get_or_insert_with(|| start.elapsed().as_secs_f64() * 1000.0);
                record.bytes += chunk.len() as u64;
            }
        }
        anyhow::ensure!(
            record.bytes == read.range.end - read.range.start,
            "Incomplete response body"
        );
        anyhow::Ok(())
    })
    .await;
    record.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            record.status = "error";
            record.error = Some(error.to_string());
        }
        Err(_) => {
            record.status = "timeout";
            record.error = Some("Request deadline exceeded".to_string());
        }
    }
    record
}

fn ranges(
    objects: &[ObjectMeta],
    size: u64,
    count: usize,
    seed: u64,
) -> anyhow::Result<Vec<(usize, Range<u64>)>> {
    let eligible = objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.size >= size)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !eligible.is_empty(),
        "No object is large enough for {size}-byte ranges"
    );
    let mut rng = StdRng::seed_from_u64(seed);
    Ok((0..count)
        .map(|ordinal| {
            let index = eligible[ordinal % eligible.len()];
            let start = rng.random_range(0..=objects[index].size - size);
            (index, start..start + size)
        })
        .collect())
}

fn emit(output: &mut impl Write, value: &impl Serialize) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)?;
    Ok(())
}

async fn collect(args: Args, store: Arc<dyn ObjectStore>, prefix: Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        !args.sizes.is_empty() && args.sizes.iter().all(|n| *n > 0),
        "Sizes must be positive"
    );
    anyhow::ensure!(
        !args.concurrency.is_empty() && args.concurrency.iter().all(|n| *n > 0),
        "Concurrency must be positive"
    );
    anyhow::ensure!(
        args.requests > 0 && args.rounds > 0,
        "Requests and rounds must be positive"
    );
    anyhow::ensure!(
        args.warmup_seconds.is_finite() && (0.0..=120.0).contains(&args.warmup_seconds),
        "Invalid warmup"
    );
    anyhow::ensure!(
        args.timeout_seconds.is_finite()
            && (0.0..=600.0).contains(&args.timeout_seconds)
            && args.timeout_seconds > 0.0,
        "Invalid request deadline"
    );
    let timeout = Duration::from_secs_f64(args.timeout_seconds);
    let epoch = Epoch {
        start: Instant::now(),
        unix_ms: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
    };
    let catalog_start = Instant::now();
    let mut objects =
        tokio::time::timeout(timeout, store.list(Some(&prefix)).try_collect::<Vec<_>>()).await??;
    let catalog_duration_ms = catalog_start.elapsed().as_secs_f64() * 1000.0;
    objects.sort_by(|a, b| a.location.cmp(&b.location));
    anyhow::ensure!(!objects.is_empty(), "No objects below {}", args.source);
    for size in &args.sizes {
        ranges(&objects, *size, 1, args.seed)?;
    }
    let system = sysinfo::System::new_all();
    let endpoint_override = [
        "AWS_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_S3_ENDPOINT",
    ]
    .iter()
    .any(|name| std::env::var_os(name).is_some());
    let mut output = BufWriter::new(
        File::options()
            .create_new(true)
            .write(true)
            .open(&args.output)
            .with_context(|| {
                format!(
                    "Create {} without overwriting earlier evidence",
                    args.output.display()
                )
            })?,
    );
    emit(
        &mut output,
        &json!({
            "kind": "environment", "schema_version": 1, "measurement_id": args.measurement_id,
            "ci_run_id": std::env::var("GITHUB_RUN_ID").ok(), "ci_run_attempt": std::env::var("GITHUB_RUN_ATTEMPT").ok(),
            "sha": std::env::var("GITHUB_SHA").ok(), "source": args.source.as_str(), "local_smoke": args.allow_local,
            "endpoint_override": endpoint_override,
            "region": std::env::var("AWS_REGION").ok(), "instance_id": std::env::var("RUNS_ON_INSTANCE_ID").ok(),
            "instance_type": std::env::var("RUNS_ON_INSTANCE_TYPE").ok(), "availability_zone": std::env::var("RUNS_ON_AWS_AZ").ok(),
            "cpus": system.cpus().len(), "total_memory": system.total_memory(), "started_unix_ms": epoch.unix_ms,
            "sizes": args.sizes, "concurrency": args.concurrency, "requests_per_cell": args.requests, "rounds": args.rounds,
            "seed": args.seed, "warmup_seconds": args.warmup_seconds, "timeout_seconds": args.timeout_seconds,
            "client": "object_store; dependency version pinned in Cargo.lock", "allocator": "mimalloc; purging disabled",
            "scope": "SDK operations; GET body fully consumed; internal HTTP retries are included in duration but not separately counted",
            "objects": objects.iter().map(|o| json!({"path": o.location.as_ref(), "size": o.size, "e_tag": o.e_tag,
                                                   "version": o.version, "last_modified": o.last_modified.to_rfc3339()})).collect::<Vec<_>>()
        }),
    )?;
    let mut catalog = Request::new("LIST", &prefix, &epoch);
    catalog.start_ms = 0.0;
    catalog.started_unix_ms = epoch.unix_ms;
    catalog.duration_ms = catalog_duration_ms;
    emit(&mut output, &catalog)?;
    for object in &objects {
        let mut request = Request::new("HEAD", &object.location, &epoch);
        let start = Instant::now();
        let meta = tokio::time::timeout(timeout, store.head(&object.location)).await??;
        request.duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        request.response_ms = Some(request.duration_ms);
        request.object_size = Some(meta.size);
        request.e_tag = meta.e_tag.clone();
        request.version = meta.version.clone();
        anyhow::ensure!(
            meta.size == object.size && meta.e_tag == object.e_tag,
            "Object changed during catalog setup"
        );
        emit(&mut output, &request)?;
        output.flush()?;
    }
    output.flush()?;
    let warmup_start = Instant::now();
    let warmup_size = objects.iter().map(|o| o.size).max().unwrap_or(0).min(65536);
    let inflight = Inflight::default();
    let mut warmup_batch = 0;
    while warmup_start.elapsed().as_secs_f64() < args.warmup_seconds {
        let reads = ranges(
            &objects,
            warmup_size,
            32,
            args.seed.wrapping_add(warmup_batch),
        )?;
        let records = stream::iter(reads.into_iter().enumerate())
            .map(|(ordinal, (index, range))| {
                measure_get(
                    store.as_ref(),
                    Read {
                        object: &objects[index],
                        range,
                        phase: "warmup",
                        round: 0,
                        ordinal,
                        concurrency: 32,
                    },
                    &epoch,
                    &inflight,
                    timeout,
                )
            })
            .buffer_unordered(32)
            .collect::<Vec<_>>()
            .await;
        for record in &records {
            emit(&mut output, record)?;
        }
        output.flush()?;
        anyhow::ensure!(
            records.iter().all(|r| r.status == "complete"),
            "Warmup request failed"
        );
        warmup_batch += 1;
    }
    let mut errors = 0;
    for round in 0..args.rounds {
        let mut cells = args
            .sizes
            .iter()
            .flat_map(|size| args.concurrency.iter().map(move |c| (*size, *c)))
            .collect::<Vec<_>>();
        cells.shuffle(&mut StdRng::seed_from_u64(
            args.seed.wrapping_add(round as u64),
        ));
        for (size, concurrency) in cells {
            // Reuse identical objects/ranges at every concurrency so load is the only changed input.
            let reads = ranges(
                &objects,
                size,
                args.requests,
                args.seed ^ size ^ round as u64,
            )?;
            let inflight = Inflight::default();
            let start = Instant::now();
            let mut records = stream::iter(reads.into_iter().enumerate())
                .map(|(ordinal, (index, range))| {
                    measure_get(
                        store.as_ref(),
                        Read {
                            object: &objects[index],
                            range,
                            phase: "measure",
                            round,
                            ordinal,
                            concurrency,
                        },
                        &epoch,
                        &inflight,
                        timeout,
                    )
                })
                .buffer_unordered(concurrency)
                .collect::<Vec<_>>()
                .await;
            let seconds = start.elapsed().as_secs_f64();
            records.sort_by_key(|r| r.ordinal);
            let failed = records.iter().filter(|r| r.status != "complete").count();
            errors += failed;
            for record in &records {
                emit(&mut output, record)?;
            }
            emit(
                &mut output,
                &json!({"kind": "cell_summary", "round": round, "range_bytes": size,
                "requested_concurrency": concurrency, "requests": records.len(), "errors": failed,
                "bytes": records.iter().map(|r| r.bytes).sum::<u64>(), "seconds": seconds,
                "peak_inflight": inflight.peak.load(Ordering::Relaxed),
                "avg_inflight": records.iter().map(|r| r.duration_ms).sum::<f64>() / (seconds * 1000.0)}),
            )?;
            output.flush()?;
            eprintln!(
                "round={round} size={size} concurrency={concurrency} requests={} errors={failed} seconds={seconds:.3}",
                records.len()
            );
        }
    }
    emit(&mut output, &json!({"kind": "complete", "errors": errors}))?;
    output.flush()?;
    anyhow::ensure!(
        errors == 0,
        "{errors} requests failed; partial measurements were preserved"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    vortex_bench::configure_allocator();
    let args = Args::parse();
    anyhow::ensure!(
        args.source.path().ends_with('/'),
        "Source prefix must end with a slash"
    );
    anyhow::ensure!(
        args.source.username().is_empty()
            && args.source.password().is_none()
            && args.source.query().is_none()
            && args.source.fragment().is_none(),
        "Source must be a plain object prefix"
    );
    let store: Arc<dyn ObjectStore> = match args.source.scheme() {
        "s3" => Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(args.source.host_str().context("Missing bucket")?)
                .build()?,
        ),
        "file" if args.allow_local => Arc::new(LocalFileSystem::new()),
        _ => anyhow::bail!("Use an s3:// prefix; filesystem smoke runs need --allow-local"),
    };
    let prefix = Path::from_url_path(args.source.path())?;
    collect(args, store, prefix).await
}

#[cfg(test)]
mod tests {
    use object_store::memory::InMemory;
    use object_store::throttle::ThrottleConfig;
    use object_store::throttle::ThrottledStore;

    #[tokio::test]
    async fn deadlines_preserve_failures_and_release_inflight_requests() -> anyhow::Result<()> {
        let store = InMemory::new();
        let path = Path::from("object");
        store.put(&path, vec![7; 1024].into()).await?;
        let object = store.head(&path).await?;
        let store = ThrottledStore::new(
            store,
            ThrottleConfig {
                wait_get_per_call: Duration::from_secs(1),
                ..ThrottleConfig::default()
            },
        );
        let epoch = Epoch {
            start: Instant::now(),
            unix_ms: 1000,
        };
        let inflight = Inflight::default();
        let record = measure_get(
            &store,
            Read {
                object: &object,
                range: 0..24,
                phase: "measure",
                round: 0,
                ordinal: 0,
                concurrency: 1,
            },
            &epoch,
            &inflight,
            Duration::from_millis(1),
        )
        .await;
        assert_eq!(record.status, "timeout");
        assert_eq!(record.bytes, 0);
        assert!(record.response_ms.is_none());
        assert_eq!(inflight.current.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[tokio::test]
    async fn reads_overlap_without_blocking_the_executor() -> anyhow::Result<()> {
        let store = InMemory::new();
        let path = Path::from("object");
        store.put(&path, vec![7; 1024].into()).await?;
        let object = store.head(&path).await?;
        let store = ThrottledStore::new(
            store,
            ThrottleConfig {
                wait_get_per_call: Duration::from_millis(20),
                ..ThrottleConfig::default()
            },
        );
        let epoch = Epoch {
            start: Instant::now(),
            unix_ms: 1000,
        };
        let inflight = Inflight::default();
        let records = futures::future::join_all((0..4).map(|ordinal| {
            measure_get(
                &store,
                Read {
                    object: &object,
                    range: 0..24,
                    phase: "measure",
                    round: 0,
                    ordinal,
                    concurrency: 4,
                },
                &epoch,
                &inflight,
                Duration::from_secs(1),
            )
        }))
        .await;
        assert!(records.iter().all(|r| r.status == "complete"));
        assert_eq!(inflight.peak.load(Ordering::Relaxed), 4);
        assert_eq!(inflight.current.load(Ordering::Relaxed), 0);
        Ok(())
    }

    use super::*;

    #[tokio::test]
    async fn measured_ranges_are_consumed_and_identified() -> anyhow::Result<()> {
        let store = InMemory::new();
        let path = Path::from("object");
        store.put(&path, vec![7; 1024].into()).await?;
        let object = store.head(&path).await?;
        let epoch = Epoch {
            start: Instant::now(),
            unix_ms: 1000,
        };
        let inflight = Inflight::default();
        let record = measure_get(
            &store,
            Read {
                object: &object,
                range: 5..29,
                phase: "measure",
                round: 1,
                ordinal: 0,
                concurrency: 4,
            },
            &epoch,
            &inflight,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(record.status, "complete");
        assert_eq!(record.bytes, 24);
        assert_eq!(record.e_tag, object.e_tag);
        assert!(record.response_ms <= record.first_byte_ms);
        assert!(
            record
                .first_byte_ms
                .is_some_and(|ms| ms <= record.duration_ms)
        );
        assert_eq!(inflight.current.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[tokio::test]
    async fn changed_object_is_invalid_instead_of_service_latency() -> anyhow::Result<()> {
        let store = InMemory::new();
        let path = Path::from("object");
        store.put(&path, vec![7; 1024].into()).await?;
        let object = store.head(&path).await?;
        store.put(&path, vec![8; 1024].into()).await?;
        let epoch = Epoch {
            start: Instant::now(),
            unix_ms: 1000,
        };
        let inflight = Inflight::default();
        let record = measure_get(
            &store,
            Read {
                object: &object,
                range: 0..24,
                phase: "measure",
                round: 0,
                ordinal: 0,
                concurrency: 1,
            },
            &epoch,
            &inflight,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(record.status, "error");
        assert_eq!(record.bytes, 0);
        assert_eq!(inflight.current.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[tokio::test]
    async fn request_ranges_are_reproducible_and_fit_the_objects() -> anyhow::Result<()> {
        let store = InMemory::new();
        store.put(&Path::from("small"), vec![0; 8].into()).await?;
        store.put(&Path::from("large"), vec![0; 128].into()).await?;
        let objects = store.list(None).try_collect::<Vec<_>>().await?;
        let first = ranges(&objects, 24, 100, 42)?;
        assert_eq!(first, ranges(&objects, 24, 100, 42)?);
        assert!(
            first
                .iter()
                .all(|(i, range)| range.end <= objects[*i].size && range.end - range.start == 24)
        );
        assert!(ranges(&objects, 129, 1, 42).is_err());
        Ok(())
    }
}
