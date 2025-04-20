use std::process::ExitCode;

use bench_vortex::bench_run::run_with_setup;
use bench_vortex::datasets::taxi_data::{taxi_data_parquet, taxi_data_vortex};
use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::TimingMeasurement;
use bench_vortex::random_access::take::{take_parquet, take_vortex_tokio};
use bench_vortex::utils::constants::STORAGE_NVME;
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{Engine, Format, default_env_filter, feature_flagged_allocator, setup_logger};
use clap::Parser;
use indicatif::ProgressBar;
use tokio::runtime::Runtime;
use vortex::buffer::{Buffer, buffer};

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Format::Parquet, Format::OnDiskVortex])]
    formats: Vec<Format>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let runtime = new_tokio_runtime(args.threads);

    let indices = buffer![10u64, 11, 12, 13, 100_000, 3_000_000];
    random_access(
        runtime,
        args.iterations,
        args.formats,
        args.display_format,
        args.verbose,
        indices,
    )
}

fn random_access(
    runtime: Runtime,
    iterations: usize,
    formats: Vec<Format>,
    display_format: DisplayFormat,
    verbose: bool,
    indices: Buffer<u64>,
) -> ExitCode {
    // Capture `RUST_LOG` configuration
    let filter = default_env_filter(verbose);
    setup_logger(filter);

    // Set up a progress bar
    let progress = ProgressBar::new(formats.len() as u64);

    let mut measurements = Vec::new();

    let taxi_vortex = runtime.block_on(taxi_data_vortex());
    let taxi_parquet = runtime.block_on(taxi_data_parquet());
    measurements.push(TimingMeasurement {
        name: "random-access/vortex-tokio-local-disk".to_string(),
        storage: STORAGE_NVME.to_owned(),
        format: Format::OnDiskVortex,
        time: run_with_setup(
            &runtime,
            iterations,
            || indices.clone(),
            |indices| async { take_vortex_tokio(&taxi_vortex, indices).await.unwrap() },
        ),
        engine: Engine::Vortex,
    });
    progress.inc(1);

    if formats.contains(&Format::Parquet) {
        measurements.push(TimingMeasurement {
            name: "random-access/parquet-tokio-local-disk".to_string(),
            storage: STORAGE_NVME.to_owned(),
            format: Format::Parquet,
            time: run_with_setup(
                &runtime,
                iterations,
                || indices.clone(),
                |indices| async { take_parquet(&taxi_parquet, indices).await.unwrap() },
            ),
            engine: Engine::Vortex,
        });
        progress.inc(1);
    }

    match display_format {
        DisplayFormat::Table => {
            render_table(measurements, &formats, RatioMode::Time, &[Engine::Vortex]).unwrap();
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements).unwrap();
        }
    }

    progress.finish();
    ExitCode::SUCCESS
}
