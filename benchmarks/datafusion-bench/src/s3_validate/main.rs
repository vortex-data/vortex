// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Paired real S3 and deterministic RAM-backed S3 measurements on one host.

mod reader;
mod store;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use clap::Parser;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::prelude::SessionContext;
use datafusion_physical_plan::collect;
use object_store::path::Path;
use serde_json::json;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::create_benchmark;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexTableOptions;

use crate::store::Model;
use crate::store::Store;

#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,
    /// Dataset root containing the format subdirectory, shared by both stores.
    #[arg(long)]
    source: String,
    #[arg(long, default_value = "10")]
    scale_factor: String,
    #[arg(long, default_value = "vortex")]
    format: Format,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    /// Per-query repetition counts, e.g. 72=3 when the default is five.
    #[arg(long, value_delimiter = ',', value_parser = iteration_override)]
    iteration_overrides: Vec<(usize, usize)>,
    #[arg(long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    /// Spread query repetitions across rounds to observe changes over time.
    #[arg(long)]
    round_robin: bool,
    /// Add an independent real S3 measurement to quantify reference repeatability.
    #[arg(long)]
    real_repeat: bool,
    /// Prime the source independently of database caches, outside query timers.
    #[arg(long, default_value_t = 0.0)]
    source_warmup_seconds: f64,
    #[arg(long, default_value_t = 25.0)]
    ttfb_ms: f64,
    #[arg(long, default_value_t = 100.0)]
    request_mib_per_sec: f64,
    /// Zero means unlimited. Applied to the model only.
    #[arg(long, default_value_t = 0)]
    max_inflight: usize,
    /// Request-derived parameters, including an optional empirical latency distribution.
    #[arg(long)]
    model_parameters: Option<PathBuf>,
    /// Optional frozen model to measure alongside the candidate and real references.
    #[arg(long)]
    baseline_model_parameters: Option<PathBuf>,
    #[arg(long, default_value_t = 20261027)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
    /// Allows a local filesystem source for development smoke runs.
    #[arg(long)]
    allow_local: bool,
    /// Retain normal coalescing, repartitioning and predicate pushdown for diagnostic runs.
    #[arg(long)]
    adaptive_io: bool,
}

fn iteration_override(value: &str) -> Result<(usize, usize), String> {
    let (query, iterations) = value.split_once('=').ok_or("Use QUERY=ITERATIONS")?;
    let query = query.parse::<usize>().map_err(|error| error.to_string())?;
    let iterations = iterations
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    if query == 0 || iterations == 0 {
        return Err("Query and iteration count must be positive".to_string());
    }
    Ok((query, iterations))
}

fn schedule<'a>(
    queries: &'a [(usize, String)],
    iterations: usize,
    overrides: &BTreeMap<usize, usize>,
    round_robin: bool,
) -> Vec<(usize, &'a str, usize)> {
    let mut measurements = queries
        .iter()
        .flat_map(|(query, sql)| {
            (0..overrides.get(query).copied().unwrap_or(iterations))
                .map(move |iteration| (*query, sql.as_str(), iteration))
        })
        .collect::<Vec<_>>();
    if round_robin {
        let positions = queries
            .iter()
            .enumerate()
            .map(|(position, (query, _))| (*query, position))
            .collect::<BTreeMap<_, _>>();
        measurements.sort_by_key(|(query, _, iteration)| {
            (
                *iteration,
                (positions[query] + queries.len() - iteration % queries.len()) % queries.len(),
            )
        });
    }
    measurements
}

fn backend_order(count: usize, query: usize, iteration: usize) -> Vec<usize> {
    let mut order = (0..count).collect::<Vec<_>>();
    if count == 4 {
        // Five is coprime with 24: visit every permutation while spreading positions over time.
        let mut rank = (iteration * 5 + query) % 24;
        let mut available = order;
        order = Vec::with_capacity(count);
        for remaining in (1..=count).rev() {
            let factorial = (1..remaining).product::<usize>();
            order.push(available.remove(rank / factorial));
            rank %= factorial;
        }
        return order;
    }
    order.rotate_left((query + iteration) % count);
    if count > 2 && (iteration / count) % 2 == 1 {
        order.reverse();
    }
    order
}

fn validate_model(model: &Model) -> anyhow::Result<()> {
    anyhow::ensure!(
        model.ttfb_ms.is_finite() && model.ttfb_ms >= 0.0,
        "invalid TTFB"
    );
    anyhow::ensure!(
        model.request_mib_per_sec.is_finite() && model.request_mib_per_sec > 0.0,
        "invalid throughput"
    );
    anyhow::ensure!(
        std::iter::once((&model.latency_cdf_ms, &model.cdf_probabilities))
            .chain(
                model
                    .latency_bins
                    .iter()
                    .map(|bin| (&bin.latency_cdf_ms, &bin.cdf_probabilities))
            )
            .all(|(cdf, probabilities)| {
                cdf.iter().all(|v| v.is_finite() && *v >= 0.0)
                    && cdf.windows(2).all(|w| w[0] <= w[1])
                    && (probabilities.is_empty()
                        || (probabilities.len() == cdf.len()
                            && probabilities
                                .iter()
                                .all(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
                            && probabilities.windows(2).all(|w| w[0] < w[1])
                            && probabilities.last() == Some(&1.0)))
            })
            && model
                .latency_bins
                .windows(2)
                .all(|bins| bins[0].max_bytes < bins[1].max_bytes),
        "latency CDF must be finite, nonnegative and sorted"
    );
    Ok(())
}

async fn session(
    benchmark: &dyn Benchmark,
    format: Format,
    store: Arc<Store>,
    adaptive_io: bool,
) -> anyhow::Result<SessionContext> {
    let session = datafusion_bench::get_session_context();
    session.register_object_store(benchmark.data_url(), Arc::<Store>::clone(&store));
    if !adaptive_io {
        session
            .sql("SET datafusion.optimizer.enable_dynamic_filter_pushdown = false")
            .await?
            .collect()
            .await?;
        session
            .sql("SET datafusion.optimizer.repartition_file_scans = false")
            .await?
            .collect()
            .await?;
    }
    for sql in benchmark.engine_init_sql(Engine::DataFusion) {
        session.sql(&sql).await?.collect().await?;
    }
    // A fresh Vortex session also clears footer and segment caches between cold runs.
    let vortex = VortexSession::default().with_tokio();
    let mut options = VortexTableOptions::default();
    options.predicate_pushdown = adaptive_io;
    options.projection_pushdown = true;
    let file_format = match format {
        Format::OnDiskVortex => {
            let mut format = VortexFormat::new_with_options(vortex, options);
            if !adaptive_io {
                format = format.with_vortex_reader_factory(Arc::new(reader::ReaderFactory(store)));
            }
            Arc::new(format) as _
        }
        Format::Parquet => datafusion_bench::format_to_df_format(format)?,
        _ => anyhow::bail!("Only vortex and parquet are supported"),
    };
    let base = benchmark.data_url().join(&format!("{}/", format.name()))?;
    for table in benchmark.table_specs() {
        let url = ListingTableUrl::try_new(base.clone(), benchmark.pattern(table.name, format))?;
        let mut config = ListingTableConfig::new(url)
            .with_listing_options(ListingOptions::new(Arc::clone(&file_format)));
        config = config.infer_schema(&session.state()).await?;
        if let Some(schema) = table.schema {
            config = config.with_schema(Arc::new(schema));
        }
        session.register_table(table.name, Arc::new(ListingTable::try_new(config)?))?;
    }
    Ok(session)
}

async fn query(session: &SessionContext, sql: &str) -> anyhow::Result<usize> {
    let frame = session.sql(sql).await?;
    let task = frame.task_ctx();
    let plan = frame.create_physical_plan().await?;
    let batches = collect(plan, Arc::new(task)).await?;
    Ok(batches.iter().map(|b| b.num_rows()).sum())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let process_start = Instant::now();
    let process_started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let args = Args::parse();
    let iteration_overrides = args
        .iteration_overrides
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    anyhow::ensure!(
        iteration_overrides.len() == args.iteration_overrides.len(),
        "Duplicate iteration override"
    );
    anyhow::ensure!(args.iterations > 0, "iterations must be positive");
    anyhow::ensure!(
        args.source_warmup_seconds.is_finite()
            && (0.0..=120.0).contains(&args.source_warmup_seconds),
        "Source warmup must be within 0–120 seconds"
    );
    let model = match &args.model_parameters {
        Some(path) => serde_json::from_reader(File::open(path)?)?,
        None => Model {
            ttfb_ms: args.ttfb_ms,
            request_mib_per_sec: args.request_mib_per_sec,
            max_inflight: args.max_inflight,
            latency_cdf_ms: Vec::new(),
            cdf_probabilities: Vec::new(),
            latency_bins: Vec::new(),
            seed: args.seed,
        },
    };
    validate_model(&model)?;
    let baseline_model: Option<Model> = args
        .baseline_model_parameters
        .as_ref()
        .map(|path| -> anyhow::Result<Model> { Ok(serde_json::from_reader(File::open(path)?)?) })
        .transpose()?;
    if let Some(baseline) = &baseline_model {
        validate_model(baseline)?;
    }
    let source_url = Url::parse(&args.source)?;
    anyhow::ensure!(
        source_url.path().ends_with('/'),
        "source must end with a slash"
    );
    anyhow::ensure!(
        source_url.scheme() == "s3" || (args.allow_local && source_url.scheme() == "file"),
        "Use S3 for validation; local sources require --allow-local"
    );
    let opts = Opts::from(vec![
        format!("scale-factor={}", args.scale_factor).parse::<Opt>()?,
        format!("remote-data-dir={}", args.source).parse::<Opt>()?,
    ]);
    let benchmark = create_benchmark(args.benchmark, &opts)?;
    let setup_session = datafusion_bench::get_session_context();
    let source = datafusion_bench::make_object_store(&setup_session, &source_url)?;
    let prefix = Path::from_url_path(source_url.join(&format!("{}/", args.format.name()))?.path())?;
    let snapshot = store::snapshot(source.as_ref(), &prefix).await?;
    let source_warmup =
        store::warm_source(source.as_ref(), &snapshot, args.source_warmup_seconds).await?;
    let objects = store::manifest(&snapshot);
    let real = Arc::new(Store::new(
        Arc::clone(&source),
        Some(Arc::clone(&snapshot)),
        None,
        !args.adaptive_io,
    ));
    let real_repeat = Arc::new(Store::new(
        Arc::clone(&source),
        Some(Arc::clone(&snapshot)),
        None,
        !args.adaptive_io,
    ));
    let baseline_simulated = baseline_model.as_ref().map(|baseline| {
        Arc::new(Store::new(
            Arc::clone(&source),
            Some(Arc::clone(&snapshot)),
            Some(baseline.clone()),
            !args.adaptive_io,
        ))
    });
    let simulated = Arc::new(Store::new(
        source,
        Some(snapshot),
        Some(model.clone()),
        !args.adaptive_io,
    ));
    let mut writer = BufWriter::new(File::create(&args.output)?);
    let system = sysinfo::System::new_all();
    serde_json::to_writer(
        &mut writer,
        &json!({"kind": "environment", "source": args.source,
        "suite": benchmark.dataset_name(), "scale_factor": args.scale_factor, "format": args.format.name(),
        "model": model, "baseline_model": baseline_model, "iterations": args.iterations, "cpus": system.cpus().len(),
        "iteration_overrides": iteration_overrides,
        "round_robin": args.round_robin, "real_repeat": args.real_repeat,
        "process_started_unix_ms": process_started_unix_ms,
        "source_warmup_seconds": args.source_warmup_seconds, "source_warmup": source_warmup,
        "availability_zone": std::env::var("RUNS_ON_AWS_AZ").ok(),
        "instance_type": std::env::var("RUNS_ON_INSTANCE_TYPE").ok(),
        "objects": objects,
        "total_memory": system.total_memory(), "available_memory": system.available_memory(),
        "ci_run_id": std::env::var("GITHUB_RUN_ID").ok(), "sha": std::env::var("GITHUB_SHA").ok(),
        "region": std::env::var("AWS_REGION").ok(), "instance_id": std::env::var("RUNS_ON_INSTANCE_ID").ok(),
        "local_smoke": args.allow_local, "adaptive_io": args.adaptive_io,
        "tail_cache_bytes": if args.adaptive_io { 0 } else { store::TAIL_CACHE_BYTES } }),
    )?;
    writeln!(writer)?;
    let queries = benchmark
        .queries()?
        .into_iter()
        .filter(|(index, _)| args.queries.as_ref().is_none_or(|qs| qs.contains(index)))
        .collect::<Vec<_>>();
    let expected = match (benchmark.dataset_name(), args.scale_factor.as_str()) {
        ("tpch", "10" | "10.0") => Some(vortex_bench::tpch::EXPECTED_ROW_COUNTS_SF10.to_vec()),
        ("tpch", "1" | "1.0") => Some(vortex_bench::tpch::EXPECTED_ROW_COUNTS_SF1.to_vec()),
        _ => None,
    };
    let mut backends = vec![("real", &real), ("model", &simulated)];
    if args.real_repeat {
        backends.push(("real_repeat", &real_repeat));
    }
    if let Some(baseline) = &baseline_simulated {
        backends.push(("model_baseline", baseline));
    }
    for (index, sql, iteration) in schedule(
        &queries,
        args.iterations,
        &iteration_overrides,
        args.round_robin,
    ) {
        let mut pair_rows = None;
        for (position, backend_index) in backend_order(backends.len(), index, iteration)
            .into_iter()
            .enumerate()
        {
            let (backend, store) = backends[backend_index];
            let measurement_model = baseline_model
                .as_ref()
                .filter(|_| backend == "model_baseline")
                .unwrap_or(&model);
            store.reset()?;
            let setup_start = Instant::now();
            let session = session(
                benchmark.as_ref(),
                args.format,
                Arc::clone(store),
                args.adaptive_io,
            )
            .await?;
            let setup_seconds = setup_start.elapsed().as_secs_f64();
            let (setup_counts, _) = store.take()?;
            let runtime = session.runtime_env();
            let caches = &runtime.cache_manager;
            caches.get_file_metadata_cache().clear();
            if let Some(cache) = caches.get_file_statistic_cache() {
                cache.clear();
            }
            for cache in ["cold", "warm"] {
                store.reset()?;
                let started_ms = process_start.elapsed().as_secs_f64() * 1000.0;
                let start = Instant::now();
                let result = query(&session, sql).await;
                let seconds = start.elapsed().as_secs_f64();
                let (counts, requests) = store.take()?;
                let (rows, error) = match result {
                    Ok(rows) => {
                        if let Some(previous) = pair_rows {
                            anyhow::ensure!(previous == rows, "Row mismatch q{index}");
                        }
                        pair_rows = Some(rows);
                        if let Some(expected) = &expected {
                            anyhow::ensure!(
                                expected[index] == rows,
                                "Unexpected row count q{index}"
                            );
                        }
                        (Some(rows), None)
                    }
                    Err(error) => (None, Some(format!("{error:#}"))),
                };
                eprintln!(
                    "{} q{index:02} {backend} {cache} iteration={iteration} {seconds:.3}s GET={} bytes={} error={}",
                    benchmark.dataset_name(),
                    counts.get,
                    counts.bytes,
                    error.as_deref().unwrap_or("none")
                );
                serde_json::to_writer(
                    &mut writer,
                    &json!({"kind": "sample", "suite": benchmark.dataset_name(),
                        "query": index, "backend": backend, "cache": cache, "iteration": iteration,
                        "position": position, "started_ms": started_ms,
                    "process_started_unix_ms": process_started_unix_ms,
                        "seconds": seconds, "rows": rows, "error": error, "counts": counts,
                        "avg_inflight": counts.request_seconds / seconds, "requests": requests,
                    "setup_seconds": setup_seconds, "setup_counts": setup_counts, "model": measurement_model }),
                )?;
                writeln!(writer)?;
                writer.flush()?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_preserve_overrides_and_visit_queries_before_repeating() {
        let queries = vec![(1, "one".to_string()), (2, "two".to_string())];
        let overrides = BTreeMap::from([(2, 1)]);
        let rounds = schedule(&queries, 3, &overrides, true);
        assert_eq!(
            rounds,
            vec![(1, "one", 0), (2, "two", 0), (1, "one", 1), (1, "one", 2)]
        );
        assert_eq!(schedule(&queries, 3, &overrides, false).len(), rounds.len());
    }

    #[test]
    fn three_backends_visit_every_order_in_six_rounds() {
        let mut orders = (0..6)
            .map(|iteration| backend_order(3, 1, iteration))
            .collect::<Vec<_>>();
        orders.sort();
        orders.dedup();
        assert_eq!(orders.len(), 6);
    }

    #[test]
    fn four_backends_visit_all_orders_in_twenty_four_rounds() {
        let mut orders = (0..24)
            .map(|iteration| backend_order(4, 1, iteration))
            .collect::<Vec<_>>();
        for position in 0..4 {
            for backend in 0..4 {
                assert_eq!(
                    orders
                        .iter()
                        .filter(|order| order[position] == backend)
                        .count(),
                    6
                );
            }
        }
        orders.sort();
        orders.dedup();
        assert_eq!(orders.len(), 24);
    }

    #[test]
    fn weighted_cdfs_require_complete_strictly_increasing_probability_mass() -> anyhow::Result<()> {
        let mut model: Model = serde_json::from_value(serde_json::json!({
            "ttfb_ms": 10.0, "request_mib_per_sec": 100.0, "max_inflight": 0,
            "latency_cdf_ms": [10.0, 20.0]
        }))?;
        for probabilities in [vec![0.5], vec![0.5, 0.5], vec![0.0, 1.0], vec![0.5, 1.1]] {
            model.cdf_probabilities = probabilities;
            assert!(validate_model(&model).is_err());
        }
        model.cdf_probabilities = vec![0.9, 1.0];
        validate_model(&model)?;
        Ok(())
    }

    #[test]
    fn query_positions_rotate_between_rounds() {
        let queries = vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string()),
        ];
        let order = schedule(&queries, 3, &BTreeMap::new(), true)
            .into_iter()
            .map(|(query, ..)| query)
            .collect::<Vec<_>>();
        assert_eq!(order, vec![1, 2, 3, 2, 3, 1, 3, 1, 2]);
    }
}
