use std::future::Future;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use bench_vortex::display::{print_measurements_json, render_table, DisplayFormat};
use bench_vortex::measurements::GenericMeasurement;
use bench_vortex::reader::{take_parquet, take_vortex_tokio};
use bench_vortex::taxi_data::{taxi_data_parquet, taxi_data_vortex};
use bench_vortex::{default_env_filter, feature_flagged_allocator, setup_logger, Format};
use clap::Parser;
use indicatif::ProgressBar;
use tokio::runtime::{Builder, Runtime};
use vortex::buffer::{buffer, Buffer};

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "10")]
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
    let runtime = match args.threads {
        Some(0) => panic!("Can't use 0 threads for runtime"),
        Some(1) => Builder::new_current_thread().enable_all().build(),
        Some(n) => Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => Builder::new_multi_thread().enable_all().build(),
    }
    .expect("Failed building the Runtime");

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

    let taxi_vortex = taxi_data_vortex();
    let taxi_parquet = taxi_data_parquet();
    measurements.push(GenericMeasurement {
        id: 0,
        name: "random-access/vortex-tokio-local-disk".to_string(),
        storage: "nvme".to_string(),
        format: Format::OnDiskVortex,
        time: run_with_setup(
            &runtime,
            iterations,
            || indices.clone(),
            |indices| async { take_vortex_tokio(&taxi_vortex, indices).await.unwrap() },
        ),
    });
    progress.inc(1);

    if formats.contains(&Format::Parquet) {
        measurements.push(GenericMeasurement {
            id: 0,
            name: "random-access/parquet-tokio-local-disk".to_string(),
            storage: "nvme".to_string(),
            format: Format::Parquet,
            time: run_with_setup(
                &runtime,
                iterations,
                || indices.clone(),
                |indices| async { take_parquet(&taxi_parquet, indices).await.unwrap() },
            ),
        });
        progress.inc(1);
    }

    match display_format {
        DisplayFormat::Table => {
            render_table(measurements, &formats).unwrap();
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements).unwrap();
        }
    }

    progress.finish();
    ExitCode::SUCCESS
}

fn run_with_setup<I, O, S, R, F>(
    runtime: &Runtime,
    iterations: usize,
    mut setup: S,
    mut routine: R,
) -> Duration
where
    S: FnMut() -> I,
    R: FnMut(I) -> F,
    F: Future<Output = O>,
{
    for _ in 0..2 {
        black_box(routine(setup()));
    }

    let mut fastest_result = Duration::from_millis(u64::MAX);
    for _ in 0..iterations {
        let state = black_box(setup());
        let elapsed = runtime.block_on(async {
            let start = Instant::now();
            let output = routine(state).await;
            let elapsed = start.elapsed();
            drop(black_box(output));
            elapsed
        });
        fastest_result = fastest_result.min(elapsed);
    }

    fastest_result
}
