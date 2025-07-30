// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bench_vortex::benchmark_driver::{DriverConfig, run_benchmark};
use bench_vortex::clickbench::{ClickBenchBenchmark, Flavor};
use bench_vortex::display::DisplayFormat;
use bench_vortex::tpcds::TpcDsBenchmark;
use bench_vortex::tpch::tpch_benchmark::TpcHBenchmark;
use bench_vortex::{Target, vortex_panic};
use clap::{Parser, Subcommand, value_parser};
use futures::executor::block_on;
use parquet::data_type::AsBytes;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process;
use vortex::error::vortex_err;

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

    #[arg(long, default_value_t = false)]
    delete_duckdb_database: bool,

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

#[derive(Parser, Debug)]
struct TpcDSArgs {
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
    setup_signal_handler()?;

    let args = Args::parse();

    let mut prof_ctl = block_on(jemalloc_pprof::PROF_CTL.as_ref().unwrap().lock());
    require_profiling_activated(&prof_ctl);

    match args.command {
        Commands::ClickBench(clickbench_args) => run_clickbench(clickbench_args),
        Commands::TpcH(tpch_args) => run_tpch(tpch_args),
        Commands::TpcDS(tpcds_args) => run_tpcds(tpcds_args),
    }?;

    let svg = prof_ctl
        .dump_flamegraph()
        .map_err(|e| vortex_err!("failed to dump flamegraph: {}", e))?;
    let mut file = File::create(format!("jemalloc_flamegraph_{}.svg", process::id()))?;
    file.write_all(svg.as_bytes())?;

    Ok(())
}

fn setup_signal_handler() -> anyhow::Result<()> {
    use signal_hook::{consts::SIGUSR1, iterator::Signals};

    let mut signals = Signals::new(&[SIGUSR1])?;

    println!(
        "kill -USR1 {} # to generate a memory profile flamegraph",
        process::id()
    );

    std::thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGUSR1 => {
                    println!("Received SIGUSR1 - generating memory profile flamegraph...");
                    if let Err(e) = generate_flamegraph() {
                        eprintln!("Failed to generate flamegraph: {}", e);
                    } else {
                        println!("Flamegraph generated successfully!");
                    }
                }
                _ => {}
            }
        }
    });

    Ok(())
}

fn generate_flamegraph() -> Result<(), Box<dyn std::error::Error>> {
    let mut prof_ctl = block_on(jemalloc_pprof::PROF_CTL.as_ref().unwrap().lock());
    require_profiling_activated(&prof_ctl);

    // Generate a unique filename with timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let filename = format!("jemalloc_profile_{}_{}.heap", process::id(), timestamp);

    let svg = prof_ctl
        .dump_flamegraph()
        .map_err(|e| vortex_err!("failed to dump flamegraph: {}", e))?;
    let mut file = File::create(filename)?;
    file.write_all(svg.as_bytes())?;
    drop(file);

    Ok(())
}

/// Checks whether jemalloc profiling is activated an returns an error response if not.
fn require_profiling_activated(prof_ctl: &jemalloc_pprof::JemallocProfCtl) {
    if !prof_ctl.activated() {
        vortex_panic!("jemalloc profiling is not activated, cannot proceed");
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
        delete_duckdb_database: args.common.delete_duckdb_database,
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
        delete_duckdb_database: args.common.delete_duckdb_database,
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

fn run_tpcds(args: TpcDSArgs) -> anyhow::Result<()> {
    // Create benchmark instance
    let benchmark = TpcDsBenchmark::new(args.scale_factor, args.common.use_remote_data_dir)?;

    // Configure driver
    let config = DriverConfig {
        targets: args.targets,
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format,
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        delete_duckdb_database: args.common.delete_duckdb_database,
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
