// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Paired real S3 and deterministic RAM-backed S3 measurements on one host.

mod reader;
mod store;

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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
    #[arg(long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(long, default_value_t = 25.0)]
    ttfb_ms: f64,
    #[arg(long, default_value_t = 100.0)]
    request_mib_per_sec: f64,
    /// Zero means unlimited. Applied to the model only.
    #[arg(long, default_value_t = 0)]
    max_inflight: usize,
    #[arg(long)]
    output: PathBuf,
    /// Allows a local filesystem source for development smoke runs.
    #[arg(long)]
    allow_local: bool,
    /// Retain normal coalescing, repartitioning and predicate pushdown for diagnostic runs.
    #[arg(long)]
    adaptive_io: bool,
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
        config = match table.schema {
            Some(schema) => config.with_schema(Arc::new(schema)),
            None => config.infer_schema(&session.state()).await?,
        };
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
    let args = Args::parse();
    anyhow::ensure!(args.iterations > 0, "iterations must be positive");
    anyhow::ensure!(
        args.ttfb_ms.is_finite() && args.ttfb_ms >= 0.0,
        "invalid TTFB"
    );
    anyhow::ensure!(
        args.request_mib_per_sec.is_finite() && args.request_mib_per_sec > 0.0,
        "invalid throughput"
    );
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
    let model = Model {
        ttfb_ms: args.ttfb_ms,
        request_mib_per_sec: args.request_mib_per_sec,
        max_inflight: args.max_inflight,
    };
    let real = Arc::new(Store::new(Arc::clone(&source), None, None));
    let simulated = Arc::new(Store::new(source, Some(snapshot), Some(model)));
    let mut writer = BufWriter::new(File::create(&args.output)?);
    let system = sysinfo::System::new_all();
    serde_json::to_writer(
        &mut writer,
        &json!({"kind": "environment", "source": args.source,
        "suite": benchmark.dataset_name(), "scale_factor": args.scale_factor, "format": args.format.name(),
        "model": model, "iterations": args.iterations, "cpus": system.cpus().len(),
        "total_memory": system.total_memory(), "available_memory": system.available_memory(),
        "ci_run_id": std::env::var("GITHUB_RUN_ID").ok(), "sha": std::env::var("GITHUB_SHA").ok(),
        "region": std::env::var("AWS_REGION").ok(), "instance_id": std::env::var("RUNS_ON_INSTANCE_ID").ok(),
        "local_smoke": args.allow_local, "adaptive_io": args.adaptive_io }),
    )?;
    writeln!(writer)?;
    let queries = benchmark.queries()?;
    let expected = match (benchmark.dataset_name(), args.scale_factor.as_str()) {
        ("tpch", "10" | "10.0") => Some(vortex_bench::tpch::EXPECTED_ROW_COUNTS_SF10.to_vec()),
        ("tpch", "1" | "1.0") => Some(vortex_bench::tpch::EXPECTED_ROW_COUNTS_SF1.to_vec()),
        _ => None,
    };
    for (index, sql) in queries {
        if args.queries.as_ref().is_some_and(|qs| !qs.contains(&index)) {
            continue;
        }
        for iteration in 0..args.iterations {
            let pair = if (index + iteration) % 2 == 0 {
                [("real", &real), ("model", &simulated)]
            } else {
                [("model", &simulated), ("real", &real)]
            };
            let mut pair_rows = None;
            for (backend, store) in pair {
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
                for cache in ["cold", "warm"] {
                    store.reset()?;
                    let start = Instant::now();
                    let result = query(&session, &sql).await;
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
                        "seconds": seconds, "rows": rows, "error": error, "counts": counts,
                        "avg_inflight": counts.request_seconds / seconds, "requests": requests,
                        "setup_seconds": setup_seconds, "setup_counts": setup_counts, "model": model }),
                    )?;
                    writeln!(writer)?;
                    writer.flush()?;
                }
            }
        }
    }
    Ok(())
}
