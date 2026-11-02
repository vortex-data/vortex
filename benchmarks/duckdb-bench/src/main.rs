// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod uring;
mod validation;

use std::fs::File;
use std::io::Write;
use std::num::NonZeroU64;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use clap::value_parser;
use duckdb_bench::DuckClient;
use mimalloc::MiMalloc;
use vortex::metrics::tracing::set_global_labels;
use vortex_bench::BenchmarkArg;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::create_benchmark;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::require_prepared_data;
use vortex_bench::runner::BenchmarkMode;
use vortex_bench::runner::SqlBenchmarkRunner;
use vortex_bench::runner::filter_queries;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::v3;

const S3_HTTP_INIT_SQL: [&str; 3] = [
    "SET http_retries = 8",
    "SET http_retry_wait_ms = 250",
    "SET http_retry_backoff = 2",
];

/// Common arguments shared across benchmarks
#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,

    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    tracing: bool,

    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(long, default_value_t = false)]
    delete_duckdb_database: bool,

    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,

    #[arg(short)]
    output_path: Option<PathBuf>,

    /// Additionally write benchmark ingest JSONL records to this path.
    #[arg(long = "ingest-jsonl")]
    ingest_output: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long, default_value = "unknown")]
    runner: String,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format))]
    formats: Vec<Format>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,

    /// Print EXPLAIN output for each query instead of running benchmarks.
    #[arg(long, default_value_t = false)]
    explain: bool,

    /// Print the selected query indices, one per line, and exit
    /// Knowledge of query ids lies only in this binary so we need
    /// orchestrator to know whan queries to run one by one.
    #[arg(long, default_value_t = false)]
    print_queries: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Whether to reuse the DuckDB connection across iterations. Helpful when profiling \
        to keep all work on the same threads"
    )]
    reuse: bool,

    /// Use the V1 Vortex scan. Pass `--scan-v1 false` to use the planner-driven V2 scan.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    scan_v1: bool,

    /// Limit DuckDB-managed query memory; this does not cap process RSS.
    #[arg(long)]
    memory_limit_bytes: Option<NonZeroU64>,

    /// Limit temporary spill files written by DuckDB.
    #[arg(long)]
    spill_limit_bytes: Option<NonZeroU64>,

    /// Evict and verify local input pages before every iteration (Linux only).
    #[arg(long)]
    cold_cache: bool,

    /// Save query results as Parquet after timing, for comparisons between backends.
    #[arg(long)]
    result_dir: Option<PathBuf>,

    /// Write per-iteration timings and I/O counters, including warmup iterations.
    #[arg(long)]
    io_jsonl: Option<PathBuf>,
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> anyhow::Result<()> {
    vortex_bench::configure_allocator();
    let args = Args::parse();
    let opts = Opts::from(args.options);

    let scan_v2 = !args.scan_v1;
    vortex_duckdb::set_scan_v2(scan_v2);

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        args.queries.as_ref(),
        args.exclude_queries.as_ref(),
    );

    if args.print_queries {
        for (query_idx, _) in &filtered_queries {
            println!("{query_idx}");
        }
        return Ok(());
    }

    if args.formats.is_empty() {
        anyhow::bail!("provide a format with --formats");
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    let io = uring::configure(scan_v2)?;
    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    anyhow::ensure!(
        !scan_v2
            || std::env::var("VORTEX_IO_BACKEND")
                .map_or(true, |backend| backend.is_empty() || backend == "legacy"),
        "Completion I/O requires the Linux io-uring build feature"
    );

    vortex_bench::forbid_data_generation();
    require_prepared_data(&*benchmark, &args.formats)?;

    let mut runner = SqlBenchmarkRunner::new(
        &*benchmark,
        Engine::DuckDB,
        args.runner.clone(),
        args.formats.clone(),
        args.track_memory,
        args.hide_progress_bar,
    )?;

    let benchmark_name = benchmark.dataset().to_string();
    let mut duckdb_init_sql = Vec::new();
    if benchmark.data_url().scheme() == "s3" {
        duckdb_init_sql.extend(S3_HTTP_INIT_SQL.map(String::from));
    }
    duckdb_init_sql.extend(benchmark.engine_init_sql(Engine::DuckDB));
    if let Some(limit) = args.memory_limit_bytes {
        duckdb_init_sql.push(format!("SET memory_limit = '{limit}B'"));
    }
    if let Some(limit) = args.spill_limit_bytes {
        duckdb_init_sql.push(format!("SET max_temp_directory_size = '{limit}B'"));
    }

    let mode = if args.explain {
        BenchmarkMode::Explain
    } else {
        BenchmarkMode::Run {
            iterations: args.iterations,
        }
    };

    let mut cold_files = Vec::new();
    if args.cold_cache {
        anyhow::ensure!(
            cfg!(target_os = "linux"),
            "Cold-cache eviction requires Linux"
        );
        let base = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Cold runs require local input files"))?;
        for format in &args.formats {
            collect_files(&base.join(format.name()), format.ext(), &mut cold_files)?;
        }
        anyhow::ensure!(!cold_files.is_empty(), "No input files found to evict");
    }

    let setup = |format| -> anyhow::Result<DuckClient> {
        let mut ctx = DuckClient::new(
            &*benchmark,
            format,
            args.delete_duckdb_database,
            args.threads,
        )?;
        ctx.set_init_sql(duckdb_init_sql.clone())?;
        ctx.register_tables(&*benchmark, format)?;
        // ClickBench uses octet_length for strings, which DuckDB names strlen.
        ctx.execute_query_result("create macro if not exists octet_length(a) as strlen(a)")?;
        Ok(ctx)
    };

    let mut io_output = args.io_jsonl.as_ref().map(File::create).transpose()?;
    runner.run_all(
        &filtered_queries,
        mode,
        &setup,
        |ctx, query_idx, format, query| {
            set_global_labels(vec![
                ("engine", "duckdb".to_string()),
                ("format", format.to_string()),
                ("benchmark_name", benchmark_name.clone()),
                ("query_idx", query_idx.to_string()),
            ]);

            // Make sure to reopen the duckdb connection between iterations
            if !args.reuse {
                ctx.reopen()?;
            }
            if args.cold_cache {
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                if let Some(io) = &io {
                    io.release_mappings();
                }
                #[cfg(target_os = "linux")]
                for file in &cold_files {
                    vortex_bench::cold_cache::evict_file(file)?;
                }
            }
            let before_bytes = io_output.as_ref().and_then(|_| process_read_bytes());
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            let before_io = io_output
                .as_ref()
                .and_then(|_| io.as_ref().map(|io| io.snapshot()));
            let (timing, result) = ctx.execute_query_result(query)?;
            if let Some(output) = &mut io_output {
                let read_bytes = before_bytes
                    .zip(process_read_bytes())
                    .map(|(before, after)| after.saturating_sub(before));
                #[allow(unused_mut)]
                let mut record = serde_json::json!({
                    "query": query_idx,
                    "format": format.name(),
                    "seconds": timing.map(|time| time.as_secs_f64()),
                    "rows": result.0.row_count(),
                    "verified_cold": args.cold_cache,
                    "proc_read_bytes": read_bytes,
                });
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                if let Some((before, io)) = before_io.zip(io.as_ref()) {
                    let after = io.snapshot();
                    record["logical_reads"] = (after.logical_reads - before.logical_reads).into();
                    record["logical_bytes"] = (after.logical_bytes - before.logical_bytes).into();
                    record["physical_reads"] =
                        (after.physical_reads - before.physical_reads).into();
                    record["physical_bytes"] =
                        (after.physical_bytes - before.physical_bytes).into();
                    record["resident_reads"] =
                        (after.resident_reads - before.resident_reads).into();
                    record["resident_bytes"] =
                        (after.resident_bytes - before.resident_bytes).into();
                }
                serde_json::to_writer(&mut *output, &record)?;
                writeln!(output)?;
            }
            Ok((timing, result))
        },
    )?;

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    if std::env::var("VORTEX_IO_STATS").as_deref() == Ok("1")
        && let Some(io) = &io
    {
        let stats = io.snapshot();
        eprintln!(
            "io_uring: logical_reads={} logical_bytes={} physical_reads={} physical_bytes={} resident_reads={} resident_bytes={}",
            stats.logical_reads,
            stats.logical_bytes,
            stats.physical_reads,
            stats.physical_bytes,
            stats.resident_reads,
            stats.resident_bytes,
        );
    }

    if !args.explain {
        if let Some(path) = args.ingest_output.as_ref() {
            v3::write_jsonl_to_path(path, &runner.v3_records())?;
        }

        let benchmark_id = format!("duckdb-{}", benchmark.dataset_name());
        let writer = create_output_writer(&args.display_format, args.output_path, &benchmark_id)?;
        runner.export_to(&args.display_format, writer)?;

        if let Some(directory) = &args.result_dir {
            for format in &args.formats {
                let ctx = setup(*format)?;
                let directory = directory.join(format.name());
                std::fs::create_dir_all(&directory)?;
                for (id, query) in &filtered_queries {
                    let path = directory.join(format!("q{id}.parquet"));
                    let path = path.to_string_lossy().replace('\'', "''");
                    let query = query.trim().trim_end_matches(';');
                    ctx.execute_query(&format!(
                        "COPY ({query}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)"
                    ))?;
                }
            }
        }
    }

    Ok(())
}

fn process_read_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/io")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes: ")?.trim().parse().ok())
}

fn collect_files(root: &Path, extension: &str, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, extension, files)?;
        } else if path.extension().is_some_and(|value| value == extension) {
            files.push(path);
        }
    }
    Ok(())
}
