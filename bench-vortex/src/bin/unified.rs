// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::path::PathBuf;

use bench_vortex::benchmark_driver::{DriverConfig, run_benchmark};
use bench_vortex::clickbench::Flavor;
use bench_vortex::clickbench_benchmark::ClickBenchBenchmark;
use bench_vortex::display::DisplayFormat;
use bench_vortex::tpch_benchmark::TpcHBenchmark;
use bench_vortex::{Engine, Format, IdempotentPath, Target};
use clap::{Parser, Subcommand, value_parser};
use url::Url;
use vortex::error::VortexExpect;

#[derive(Parser, Debug)]
#[command(version, about = "Unified Vortex benchmark runner", long_about = None)]
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
}

/// Common arguments shared across benchmarks
#[derive(Parser, Debug)]
struct CommonArgs {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    export_spans: bool,

    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,

    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,

    #[arg(short)]
    output_path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    show_metrics: bool,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long)]
    use_remote_data_dir: Option<String>,

    #[arg(long, default_value_t = false)]
    emit_plan: bool,
}

#[derive(Parser, Debug)]
struct ClickBenchArgs {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,

    #[arg(long)]
    queries_file: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Flavor::Partitioned)]
    flavor: Flavor,

    #[arg(long, default_value_t = false)]
    single_file: bool,
}

#[derive(Parser, Debug)]
struct TpcHArgs {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:arrow",
            "datafusion:parquet",
            "datafusion:vortex",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,

    #[arg(long, default_value_t = 1, value_parser=validate_scale_factor)]
    scale_factor: u32,
}

fn validate_scale_factor(val: &str) -> Result<u32, String> {
    match val.parse::<u32>() {
        Ok(n) if [1, 10, 100, 1000].contains(&n) => Ok(n),
        _ => Err(String::from(
            "Value must be a scale factor of 1, 10, 100 or 1000",
        )),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::ClickBench(clickbench_args) => run_clickbench(clickbench_args),
        Commands::TpcH(tpch_args) => run_tpch(tpch_args),
    }
}

fn run_clickbench(args: ClickBenchArgs) -> anyhow::Result<()> {
    // Create benchmark instance
    let benchmark = ClickBenchBenchmark::new(
        args.flavor,
        args.single_file,
        args.queries_file.map(|p| p.to_string_lossy().to_string()),
        args.common.use_remote_data_dir,
    );

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format,
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        queries: args.common.queries,
        exclude_queries: args.common.exclude_queries,
        output_path: args.common.output_path,
        emit_plan: args.common.emit_plan,
        export_spans: args.common.export_spans,
        show_metrics: args.common.show_metrics,
        hide_progress_bar: args.common.hide_progress_bar,
    };

    // Determine data URL
    let data_url = data_source_base_url(&benchmark.use_remote_data_dir, benchmark.flavor)?;

    // Run benchmark using the trait system
    run_benchmark(benchmark, config, "clickbench.trace.json", data_url)
}

fn run_tpch(args: TpcHArgs) -> anyhow::Result<()> {
    // Store needed values before they're moved
    let has_duckdb_vortex = args
        .targets
        .iter()
        .any(|t| t.engine() == Engine::DuckDB && t.format() == Format::OnDiskVortex);
    let queries_for_verify = args.common.queries.clone();

    // Create benchmark instance
    let benchmark = TpcHBenchmark::new(args.scale_factor, args.common.use_remote_data_dir);

    // Determine data URL
    let data_url = match &benchmark.use_remote_data_dir {
        None => {
            let data_dir = "tpch".to_data_path();
            let data_dir = data_dir.to_str().vortex_expect("path must be utf8");
            Url::parse(format!("file:{data_dir}/{}/", args.scale_factor).as_ref())?
        }
        Some(remote_data_dir) => Url::parse(remote_data_dir)?,
    };

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format,
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        queries: args.common.queries,
        exclude_queries: args.common.exclude_queries,
        output_path: args.common.output_path,
        emit_plan: args.common.emit_plan,
        export_spans: args.common.export_spans,
        show_metrics: args.common.show_metrics,
        hide_progress_bar: args.common.hide_progress_bar,
    };

    // Run benchmark using the trait system
    run_benchmark(benchmark, config, "tpch.trace.json", data_url.clone())?;

    // The CI env var is defined by Github Actions.
    // https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#default-environment-variables
    if has_duckdb_vortex && env::var("CI").is_ok() {
        // Re-create benchmark instance for verification
        let verify_benchmark = TpcHBenchmark::new(args.scale_factor, None);
        verify_benchmark.verify_duckdb_tpch_results(&data_url, queries_for_verify)?;
    }

    Ok(())
}

fn data_source_base_url(remote_data_dir: &Option<String>, flavor: Flavor) -> anyhow::Result<Url> {
    match remote_data_dir {
        None => {
            let basepath = format!("clickbench_{flavor}").to_data_path();
            Ok(Url::parse(&format!(
                "file:{}/",
                basepath.to_str().vortex_expect("path should be utf8")
            ))?)
        }
        Some(remote_data_dir) => {
            if !remote_data_dir.ends_with("/") {
                log::warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            log::info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\\n",
                    "--use-remote-data-dir) and upload data/clickbench/ to some remote location.",
                ),
                remote_data_dir,
            );
            Ok(Url::parse(remote_data_dir)?)
        }
    }
}
