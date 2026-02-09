// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;
use indicatif::ProgressBar;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Target;
use vortex_bench::create_output_writer;
use vortex_bench::datasets::gharchive::gharchive_parquet;
use vortex_bench::datasets::gharchive::gharchive_vortex;
use vortex_bench::datasets::taxi_data::*;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::display::render_table;
use vortex_bench::measurements::TimingMeasurement;
use vortex_bench::random_access::FieldPath;
use vortex_bench::random_access::ParquetProjectingAccessor;
use vortex_bench::random_access::ParquetRandomAccessor;
use vortex_bench::random_access::ProjectingRandomAccessor;
use vortex_bench::random_access::RandomAccessor;
use vortex_bench::random_access::VortexProjectingAccessor;
use vortex_bench::random_access::VortexRandomAccessor;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::utils::constants::STORAGE_NVME;

/// Available datasets for random access benchmarks.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Dataset {
    /// NYC Taxi trip data - flat schema with many columns.
    #[default]
    Taxi,
    /// GitHub Archive event data - deeply nested schema with struct fields.
    GhArchive,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![Format::Parquet, Format::OnDiskVortex]
    )]
    formats: Vec<Format>,
    /// Dataset to benchmark.
    #[arg(long, value_enum, default_value_t = Dataset::Taxi)]
    dataset: Dataset,
    /// Time limit in seconds for each benchmark target (e.g., 10 for 10 seconds).
    #[arg(long, default_value_t = 10)]
    time_limit: u64,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short)]
    output_path: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    match args.dataset {
        Dataset::Taxi => {
            // Row count of the taxi dataset is 3,339,715.
            let indices = vec![10u64, 11, 12, 13, 100_000, 3_000_000];
            run_taxi_random_access(
                args.formats,
                args.time_limit,
                args.display_format,
                indices,
                args.output_path,
            )
            .await
        }
        Dataset::GhArchive => {
            // Run gharchive benchmark with nested field projection.
            // The field path is payload.ref - a deeply nested string field.
            let field_path = vec!["actor".to_string(), "login".to_string()];
            // Use smaller indices as gharchive may have fewer rows per row group.
            let indices = vec![10u64, 11, 12, 13, 1_000, 10_000];
            run_gharchive_random_access(
                args.formats,
                args.time_limit,
                args.display_format,
                indices,
                field_path,
                args.output_path,
            )
            .await
        }
    }
}

/// Create a random accessor for the given format using taxi data.
async fn get_taxi_accessor(format: Format) -> anyhow::Result<Box<dyn RandomAccessor>> {
    match format {
        Format::OnDiskVortex => {
            let path = taxi_data_vortex().await?;
            Ok(Box::new(VortexRandomAccessor::new(path)))
        }
        Format::VortexCompact => {
            let path = taxi_data_vortex_compact().await?;
            Ok(Box::new(VortexRandomAccessor::compact(path)))
        }
        Format::Parquet => {
            let path = taxi_data_parquet().await?;
            Ok(Box::new(ParquetRandomAccessor::new(path)))
        }
        #[cfg(feature = "lance")]
        Format::Lance => {
            use lance_bench::random_access::LanceRandomAccessor;
            use lance_bench::random_access::taxi_data_lance;

            let path = taxi_data_lance().await?;
            Ok(Box::new(LanceRandomAccessor::new(path)))
        }
        _ => unimplemented!("Random access bench not implemented for {format}"),
    }
}

/// Create a projecting random accessor for the given format using gharchive data.
async fn get_gharchive_accessor(
    format: Format,
) -> anyhow::Result<Box<dyn ProjectingRandomAccessor>> {
    match format {
        Format::OnDiskVortex => {
            let path = gharchive_vortex().await?;
            Ok(Box::new(VortexProjectingAccessor::new(path)))
        }
        Format::VortexCompact => {
            // For now, use the same path as OnDiskVortex (compact not yet implemented for gharchive)
            let path = gharchive_vortex().await?;
            Ok(Box::new(VortexProjectingAccessor::compact(path)))
        }
        Format::Parquet => {
            let path = gharchive_parquet().await?;
            Ok(Box::new(ParquetProjectingAccessor::new(path)))
        }
        _ => unimplemented!("Projected random access bench not implemented for {format}"),
    }
}

/// Run a random access benchmark for the given accessor.
///
/// Runs the take operation repeatedly until the time limit is reached,
/// collecting timing for each run.
async fn benchmark_random_access(
    accessor: &dyn RandomAccessor,
    indices: &[u64],
    time_limit_secs: u64,
    storage: &str,
) -> anyhow::Result<TimingMeasurement> {
    let time_limit = Duration::from_secs(time_limit_secs);
    let overall_start = Instant::now();
    let mut runs = Vec::new();

    // Run at least once, then continue until time limit
    loop {
        let indices = indices.to_vec();
        let start = Instant::now();
        let _row_count = accessor.take(indices).await?;
        runs.push(start.elapsed());

        if overall_start.elapsed() >= time_limit {
            break;
        }
    }

    Ok(TimingMeasurement {
        name: accessor.name().to_string(),
        storage: storage.to_string(),
        target: Target::new(format_to_engine(accessor.format()), accessor.format()),
        runs,
    })
}

/// Run a projected random access benchmark for the given accessor.
///
/// Runs the take operation with field projection repeatedly until the time limit is reached,
/// collecting timing for each run.
async fn benchmark_projected_random_access(
    accessor: &dyn ProjectingRandomAccessor,
    indices: &[u64],
    field_path: &FieldPath,
    time_limit_secs: u64,
    storage: &str,
) -> anyhow::Result<TimingMeasurement> {
    let time_limit = Duration::from_secs(time_limit_secs);
    let overall_start = Instant::now();
    let mut runs = Vec::new();

    // Run at least once, then continue until time limit
    loop {
        let indices = indices.to_vec();
        let start = Instant::now();
        let _row_count = accessor.take_projected(indices, field_path).await?;
        runs.push(start.elapsed());

        if overall_start.elapsed() >= time_limit {
            break;
        }
    }

    Ok(TimingMeasurement {
        name: accessor.name().to_string(),
        storage: storage.to_string(),
        target: Target::new(format_to_engine(accessor.format()), accessor.format()),
        runs,
    })
}

/// Map format to the appropriate engine for random access benchmarks.
fn format_to_engine(format: Format) -> Engine {
    match format {
        Format::OnDiskVortex | Format::VortexCompact => Engine::Vortex,
        Format::Parquet => Engine::Arrow,
        #[cfg(feature = "lance")]
        Format::Lance => Engine::Arrow, // Is this right here?
        _ => Engine::default(),
    }
}

/// The benchmark ID used for output path.
const TAXI_BENCHMARK_ID: &str = "random-access-taxi";
const GHARCHIVE_BENCHMARK_ID: &str = "random-access-gharchive";

async fn run_taxi_random_access(
    formats: Vec<Format>,
    time_limit: u64,
    display_format: DisplayFormat,
    indices: Vec<u64>,
    output_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    let progress = ProgressBar::new(formats.len() as u64);

    let mut targets = Vec::new();
    let mut measurements = Vec::new();

    for format in formats {
        let accessor = get_taxi_accessor(format).await?;
        let measurement =
            benchmark_random_access(accessor.as_ref(), &indices, time_limit, STORAGE_NVME).await?;

        targets.push(measurement.target);
        measurements.push(measurement);

        progress.inc(1);
    }

    progress.finish();

    let mut writer = create_output_writer(&display_format, output_path, TAXI_BENCHMARK_ID)?;

    match display_format {
        DisplayFormat::Table => {
            render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements)?;
        }
    }

    Ok(())
}

async fn run_gharchive_random_access(
    formats: Vec<Format>,
    time_limit: u64,
    display_format: DisplayFormat,
    indices: Vec<u64>,
    field_path: FieldPath,
    output_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    let progress = ProgressBar::new(formats.len() as u64);

    let mut targets = Vec::new();
    let mut measurements = Vec::new();

    for format in formats {
        let accessor = get_gharchive_accessor(format).await?;
        let measurement = benchmark_projected_random_access(
            accessor.as_ref(),
            &indices,
            &field_path,
            time_limit,
            STORAGE_NVME,
        )
        .await?;

        targets.push(measurement.target);
        measurements.push(measurement);

        progress.inc(1);
    }

    progress.finish();

    let mut writer = create_output_writer(&display_format, output_path, GHARCHIVE_BENCHMARK_ID)?;

    match display_format {
        DisplayFormat::Table => {
            render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements)?;
        }
    }

    Ok(())
}
