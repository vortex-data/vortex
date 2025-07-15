// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use bench_vortex::Target;
use bench_vortex::benchmark_driver::{DriverConfig, run_benchmark};
use bench_vortex::clickbench::Flavor;
use bench_vortex::clickbench_benchmark::ClickBenchBenchmark;
use bench_vortex::display::DisplayFormat;
use bench_vortex::tpch_benchmark::TpcHBenchmark;
use clap::{Parser, Subcommand, value_parser};

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

    /// TODO(joe): remove this flag and add a cache flag to common.
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

    #[arg(long, default_value_t = false)]
    track_memory: bool,
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

    #[arg(long, default_value = "1.0", value_parser=validate_scale_factor)]
    scale_factor: String,
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
    }
}

fn run_clickbench(args: ClickBenchArgs) -> anyhow::Result<()> {
    // Create benchmark instance
    let benchmark = ClickBenchBenchmark::new(
        args.flavor,
        args.single_file,
        args.queries_file.map(|p| p.to_string_lossy().to_string()),
        args.common.use_remote_data_dir,
    )?;

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
        track_memory: args.common.track_memory,
    };

    // Determine data URL
    // Run benchmark using the trait system
    run_benchmark(benchmark, config)
}

fn run_tpch(args: TpcHArgs) -> anyhow::Result<()> {
    // Create benchmark instance
    let benchmark = TpcHBenchmark::new(args.scale_factor, args.common.use_remote_data_dir)?;

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
        track_memory: args.common.track_memory,
    };

    // Run benchmark using the trait system
    run_benchmark(benchmark, config)?;

    Ok(())
}
