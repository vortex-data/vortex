// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use bench_vortex::benchmark_driver::{DriverConfig, run_benchmark};
use bench_vortex::clickbench::{ClickBenchBenchmark, Flavor};
use bench_vortex::display::DisplayFormat;
use bench_vortex::fineweb::Fineweb;
use bench_vortex::realnest::gharchive::GithubArchive;
use bench_vortex::statpopgen::StatPopGenBenchmark;
use bench_vortex::tpcds::TpcDsBenchmark;
use bench_vortex::tpch::tpch_benchmark::TpcHBenchmark;
use bench_vortex::{IdempotentPath as _, Target, setup_logging_and_tracing};
use clap::{Parser, Subcommand, value_parser};
use url::Url;

#[derive(Parser, Debug)]
#[command(version, about = "Vortex query benchmark runner", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run ClickBench queries
    #[command(name = "clickbench")]
    ClickBench(ClickBenchArgs),

    /// Run TPC-H queries
    #[command(name = "tpch")]
    TpcH(TpcHArgs),

    /// Run TPC-DS queries
    #[command(name = "tpcds")]
    TpcDS(TpcDSArgs),

    /// Run Statisical & Population Genetics queries
    #[command(name = "statpopgen")]
    StatPopGen(StatPopGenArgs),

    #[command(name = "fineweb")]
    Fineweb(FinewebArgs),

    #[command(name = "gharchive")]
    GhArchive(GhArchiveArgs),
}

/// Core execution arguments - used by ALL benchmarks
#[derive(Parser, Debug)]
struct CoreArgs {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    tracing: bool,
}

/// Query filtering arguments - used by benchmarks with multiple queries
#[derive(Parser, Debug)]
struct QueryFilterArgs {
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,
}

/// Output configuration arguments
#[derive(Parser, Debug)]
struct OutputArgs {
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(short)]
    output_path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,
}

/// Engine configuration arguments
#[derive(Parser, Debug)]
struct EngineArgs {
    /// TODO(joe): remove this flag and add a cache flag to common.
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,

    #[arg(long, default_value_t = false)]
    delete_duckdb_database: bool,
}

/// Debugging/analysis arguments
#[derive(Parser, Debug)]
struct DebugArgs {
    #[arg(long)]
    export_spans: bool,

    #[arg(long, default_value_t = false)]
    show_metrics: bool,

    #[arg(long, default_value_t = false)]
    emit_plan: bool,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value_t = false)]
    explain: bool,

    #[arg(long, default_value_t = false)]
    explain_analyze: bool,
}

/// Data generation arguments
#[derive(Parser, Debug)]
struct DataArgs {
    #[arg(long, default_value_t = false)]
    skip_generate: bool,
}

/// Remote data configuration (only for benchmarks that support it)
#[derive(Parser, Debug)]
struct RemoteDataArgs {
    #[arg(long)]
    use_remote_data_dir: Option<String>,
}

#[derive(Parser, Debug)]
struct ClickBenchArgs {
    #[command(flatten)]
    core: CoreArgs,

    #[command(flatten)]
    query_filter: QueryFilterArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    debug: DebugArgs,

    #[command(flatten)]
    data: DataArgs,

    #[command(flatten)]
    remote_data: RemoteDataArgs,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "datafusion:vortex-compact",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:vortex-compact",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,

    #[arg(long)]
    queries_file: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Flavor::Partitioned)]
    flavor: Flavor,
}

#[derive(Parser, Debug)]
struct TpcHArgs {
    #[command(flatten)]
    core: CoreArgs,

    #[command(flatten)]
    query_filter: QueryFilterArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    debug: DebugArgs,

    #[command(flatten)]
    data: DataArgs,

    #[command(flatten)]
    remote_data: RemoteDataArgs,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:arrow",
            "datafusion:parquet",
            "datafusion:vortex",
            "datafusion:vortex-compact",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:vortex-compact",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,

    #[arg(long, default_value = "1.0", value_parser=validate_scale_factor)]
    scale_factor: String,
}

#[derive(Parser, Debug)]
struct TpcDSArgs {
    #[command(flatten)]
    core: CoreArgs,

    #[command(flatten)]
    query_filter: QueryFilterArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    debug: DebugArgs,

    #[command(flatten)]
    data: DataArgs,

    #[command(flatten)]
    remote_data: RemoteDataArgs,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "datafusion:vortex-compact",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:vortex-compact",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,

    #[arg(long, default_value = "1.0", value_parser=validate_scale_factor)]
    scale_factor: String,
}

#[derive(Parser, Debug)]
struct StatPopGenArgs {
    #[command(flatten)]
    core: CoreArgs,

    #[command(flatten)]
    query_filter: QueryFilterArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    debug: DebugArgs,

    #[command(flatten)]
    data: DataArgs,

    // Note: No remote_data - this benchmark doesn't support use_remote_data_dir
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
          default_values = vec![
              // DataFusion does not support list_aggregate and simulating it with an UNNEST and GROUP
              // BY is _very_ slow.
              //
              // "datafusion:parquet",
              // "datafusion:vortex",
              "duckdb:parquet",
              "duckdb:vortex",
              "duckdb:vortex-compact",
              //
              // DuckDB native has a fixed parallelism row group size of 122,880
              // rows. Unfortunately, this kind of list-heavy dataset is almost perfectly
              // adversarial to that limitation.
              //
              // https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads.html#the-effect-of-row-groups-on-parallelism
              //
              // "duckdb:duckdb"
          ]
    )]
    targets: Vec<Target>,

    #[arg(long)]
    scale_factor: u64,
}

#[derive(Parser, Debug)]
struct FinewebArgs {
    #[command(flatten)]
    core: CoreArgs,

    #[command(flatten)]
    query_filter: QueryFilterArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    debug: DebugArgs,

    #[command(flatten)]
    data: DataArgs,

    #[command(flatten)]
    remote_data: RemoteDataArgs,

    // Note: No scale_factor - this benchmark doesn't support scale_factor
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
              "duckdb:parquet",
              "duckdb:vortex",
              "duckdb:vortex-compact",
              "datafusion:parquet",
              "datafusion:vortex",
              "datafusion:vortex-compact",
          ]
    )]
    targets: Vec<Target>,
}

#[derive(Parser, Debug)]
struct GhArchiveArgs {
    #[command(flatten)]
    core: CoreArgs,

    #[command(flatten)]
    query_filter: QueryFilterArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    debug: DebugArgs,

    #[command(flatten)]
    data: DataArgs,

    #[command(flatten)]
    remote_data: RemoteDataArgs,

    // Note: No scale_factor - this benchmark doesn't support scale_factor
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
              "duckdb:parquet",
              "duckdb:vortex",
              "duckdb:vortex-compact",
              "datafusion:parquet",
              "datafusion:vortex",
              "datafusion:vortex-compact",
          ]
    )]
    targets: Vec<Target>,
}

fn validate_scale_factor(val: &str) -> Result<String, String> {
    match val.parse::<f32>() {
        Ok(n) if [0.01, 0.1, 1., 10., 100., 1000.].contains(&n) => {
            // Normalize to full decimal format
            let normalized = match n {
                0.01 => "0.01",
                0.1 => "0.1",
                1.0 => "1.0",
                10.0 => "10.0",
                100.0 => "100.0",
                1000.0 => "1000.0",
                _ => unreachable!(), // Already validated above
            };
            Ok(normalized.to_string())
        }
        _ => Err(String::from(
            "Value must be a scale factor of 0.01, 0.1, 1, 10, 100 or 1000",
        )),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Commands::ClickBench(clickbench_args) => run_clickbench(clickbench_args),
        Commands::TpcH(tpch_args) => run_tpch(tpch_args),
        Commands::TpcDS(tpcds_args) => run_tpcds(tpcds_args),
        Commands::StatPopGen(stat_pop_gen_args) => run_statpopgen(stat_pop_gen_args),
        Commands::Fineweb(fineweb_args) => run_fineweb(fineweb_args),
        Commands::GhArchive(gh_archive_args) => run_gharchive(gh_archive_args),
    }
}

fn run_clickbench(args: ClickBenchArgs) -> anyhow::Result<()> {
    setup_logging_and_tracing(args.core.verbose, args.core.tracing)?;

    // Create benchmark instance
    let benchmark = ClickBenchBenchmark::new(
        args.flavor,
        args.queries_file.map(|p| p.to_string_lossy().to_string()),
        args.remote_data.use_remote_data_dir,
    )?;

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.core.iterations,
        threads: args.core.threads,
        display_format: args.output.display_format,
        disable_datafusion_cache: args.engine.disable_datafusion_cache,
        delete_duckdb_database: args.engine.delete_duckdb_database,
        queries: args.query_filter.queries,
        exclude_queries: args.query_filter.exclude_queries,
        output_path: args.output.output_path,
        emit_plan: args.debug.emit_plan,
        export_spans: args.debug.export_spans,
        show_metrics: args.debug.show_metrics,
        hide_progress_bar: args.output.hide_progress_bar,
        track_memory: args.debug.track_memory,
        skip_generate: args.data.skip_generate,
        explain: args.debug.explain,
        explain_analyze: args.debug.explain_analyze,
    };

    // Run benchmark using the trait system
    run_benchmark(benchmark, config)
}

fn run_tpch(args: TpcHArgs) -> anyhow::Result<()> {
    setup_logging_and_tracing(args.core.verbose, args.core.tracing)?;

    // Create benchmark instance
    let benchmark = TpcHBenchmark::new(args.scale_factor, args.remote_data.use_remote_data_dir)?;

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.core.iterations,
        threads: args.core.threads,
        display_format: args.output.display_format,
        disable_datafusion_cache: args.engine.disable_datafusion_cache,
        delete_duckdb_database: args.engine.delete_duckdb_database,
        queries: args.query_filter.queries,
        exclude_queries: args.query_filter.exclude_queries,
        output_path: args.output.output_path,
        emit_plan: args.debug.emit_plan,
        export_spans: args.debug.export_spans,
        show_metrics: args.debug.show_metrics,
        hide_progress_bar: args.output.hide_progress_bar,
        track_memory: args.debug.track_memory,
        skip_generate: args.data.skip_generate,
        explain: args.debug.explain,
        explain_analyze: args.debug.explain_analyze,
    };

    // Run benchmark using the trait system
    run_benchmark(benchmark, config)?;

    Ok(())
}

fn run_tpcds(args: TpcDSArgs) -> anyhow::Result<()> {
    setup_logging_and_tracing(args.core.verbose, args.core.tracing)?;

    // Create benchmark instance
    let benchmark = TpcDsBenchmark::new(args.scale_factor, args.remote_data.use_remote_data_dir)?;

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.core.iterations,
        threads: args.core.threads,
        display_format: args.output.display_format,
        disable_datafusion_cache: args.engine.disable_datafusion_cache,
        delete_duckdb_database: args.engine.delete_duckdb_database,
        queries: args.query_filter.queries,
        exclude_queries: args.query_filter.exclude_queries,
        output_path: args.output.output_path,
        emit_plan: args.debug.emit_plan,
        export_spans: args.debug.export_spans,
        show_metrics: args.debug.show_metrics,
        hide_progress_bar: args.output.hide_progress_bar,
        track_memory: args.debug.track_memory,
        skip_generate: args.data.skip_generate,
        explain: args.debug.explain,
        explain_analyze: args.debug.explain_analyze,
    };

    // Run benchmark using the trait system
    run_benchmark(benchmark, config)?;

    Ok(())
}

fn run_statpopgen(args: StatPopGenArgs) -> anyhow::Result<()> {
    setup_logging_and_tracing(args.core.verbose, args.core.tracing)?;

    // Create benchmark instance
    let data_url = Url::from_directory_path("statpopgen".to_data_path())
        .map_err(|_| anyhow::anyhow!("bad data path?"))?;
    let benchmark = StatPopGenBenchmark::new(data_url, args.scale_factor)?;

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.core.iterations,
        threads: args.core.threads,
        display_format: args.output.display_format,
        disable_datafusion_cache: args.engine.disable_datafusion_cache,
        delete_duckdb_database: args.engine.delete_duckdb_database,
        queries: args.query_filter.queries,
        exclude_queries: args.query_filter.exclude_queries,
        output_path: args.output.output_path,
        emit_plan: args.debug.emit_plan,
        export_spans: args.debug.export_spans,
        show_metrics: args.debug.show_metrics,
        hide_progress_bar: args.output.hide_progress_bar,
        track_memory: args.debug.track_memory,
        skip_generate: args.data.skip_generate,
        explain: args.debug.explain,
        explain_analyze: args.debug.explain_analyze,
    };

    // Run benchmark using the trait system
    run_benchmark(benchmark, config)
}

fn run_fineweb(args: FinewebArgs) -> anyhow::Result<()> {
    setup_logging_and_tracing(args.core.verbose, args.core.tracing)?;

    let benchmark = Fineweb::with_remote_data_dir(args.remote_data.use_remote_data_dir)?;

    let config = DriverConfig {
        targets: args.targets,
        iterations: args.core.iterations,
        threads: args.core.threads,
        display_format: args.output.display_format,
        disable_datafusion_cache: args.engine.disable_datafusion_cache,
        delete_duckdb_database: args.engine.delete_duckdb_database,
        queries: args.query_filter.queries,
        exclude_queries: args.query_filter.exclude_queries,
        output_path: args.output.output_path,
        emit_plan: args.debug.emit_plan,
        export_spans: args.debug.export_spans,
        show_metrics: args.debug.show_metrics,
        hide_progress_bar: args.output.hide_progress_bar,
        track_memory: args.debug.track_memory,
        skip_generate: args.data.skip_generate,
        explain: args.debug.explain,
        explain_analyze: args.debug.explain_analyze,
    };

    run_benchmark(benchmark, config)
}

fn run_gharchive(args: GhArchiveArgs) -> anyhow::Result<()> {
    setup_logging_and_tracing(args.core.verbose, args.core.tracing)?;

    let benchmark = GithubArchive::with_remote_data_dir(args.remote_data.use_remote_data_dir)?;

    let config = DriverConfig {
        targets: args.targets,
        iterations: args.core.iterations,
        threads: args.core.threads,
        display_format: args.output.display_format,
        disable_datafusion_cache: args.engine.disable_datafusion_cache,
        delete_duckdb_database: args.engine.delete_duckdb_database,
        queries: args.query_filter.queries,
        exclude_queries: args.query_filter.exclude_queries,
        output_path: args.output.output_path,
        emit_plan: args.debug.emit_plan,
        export_spans: args.debug.export_spans,
        show_metrics: args.debug.show_metrics,
        hide_progress_bar: args.output.hide_progress_bar,
        track_memory: args.debug.track_memory,
        skip_generate: args.data.skip_generate,
        explain: args.debug.explain,
        explain_analyze: args.debug.explain_analyze,
    };

    run_benchmark(benchmark, config)
}
