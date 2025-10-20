// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::{Write, stdout};
use std::path::PathBuf;

use bench_vortex::bench_run::run_with_setup;
use bench_vortex::datasets::taxi_data::*;
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::measurements::TimingMeasurement;
#[cfg(feature = "lance")]
use bench_vortex::random_access::take::take_lance;
use bench_vortex::random_access::take::{take_parquet, take_vortex_tokio};
use bench_vortex::utils::constants::STORAGE_NVME;
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{Engine, Format, Target, setup_logging_and_tracing};
use clap::Parser;
use indicatif::ProgressBar;
use tokio::runtime::Runtime;
use vortex::buffer::{Buffer, buffer};
use vortex::dtype::Nullability::NonNullable;
use vortex::error::VortexExpect;
use vortex::scalar::Scalar;
use vortex::{Array, ArrayRef, ToCanonical};

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
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short)]
    output_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let runtime = new_tokio_runtime(args.threads)?;

    // Row count of the dataset is 3,339,715.
    let indices = buffer![10u64, 11, 12, 13, 100_000, 3_000_000];

    random_access(
        args.formats,
        runtime,
        args.iterations,
        args.display_format,
        indices,
        &args.output_path,
    )
}

/// Given a benchmark future, runs it and returns a [`TimingMeasurement`].
fn create_timing_measurement<O, B, F>(
    benchmark: B,
    name: String,
    storage: String,
    runtime: &Runtime,
    indices: &Buffer<u64>,
    iterations: usize,
    target: Target,
) -> TimingMeasurement
where
    B: FnMut(Buffer<u64>) -> F,
    F: Future<Output = O>,
{
    let benchmark_duration = run_with_setup(runtime, iterations, || indices.clone(), benchmark);

    TimingMeasurement {
        name,
        storage,
        target,
        time: benchmark_duration,
    }
}

fn random_access(
    formats: Vec<Format>,
    runtime: Runtime,
    iterations: usize,
    display_format: DisplayFormat,
    indices: Buffer<u64>,
    output_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    let progress = ProgressBar::new(formats.len() as u64);

    let mut targets = Vec::new();
    let mut measurements = Vec::new();

    for format in formats {
        let engine = match format {
            Format::OnDiskVortex | Format::VortexCompact => Engine::Vortex,
            Format::Parquet => Engine::Arrow,
            #[cfg(feature = "lance")]
            Format::Lance => Engine::Arrow,
            Format::Csv | Format::Arrow | Format::OnDiskDuckDB => unimplemented!(),
        };
        let target = Target::new(engine, format);

        let timing_measurement = match format {
            Format::OnDiskVortex => {
                let taxi_vortex = runtime.block_on(taxi_data_vortex())?;

                create_timing_measurement(
                    |indices| async {
                        take_vortex_tokio(&taxi_vortex, indices, validate_vortex_array).await
                    },
                    "random-access/vortex-tokio-local-disk".to_string(),
                    STORAGE_NVME.to_owned(),
                    &runtime,
                    &indices,
                    iterations,
                    target,
                )
            }
            Format::VortexCompact => {
                let taxi_vortex_compact = runtime.block_on(taxi_data_vortex_compact())?;

                create_timing_measurement(
                    |indices| async {
                        take_vortex_tokio(&taxi_vortex_compact, indices, validate_vortex_array)
                            .await
                    },
                    "random-access/vortex-compact-tokio-local-disk".to_string(),
                    STORAGE_NVME.to_owned(),
                    &runtime,
                    &indices,
                    iterations,
                    target,
                )
            }
            Format::Parquet => {
                let taxi_parquet = runtime.block_on(taxi_data_parquet())?;

                create_timing_measurement(
                    |indices| async { take_parquet(&taxi_parquet, indices).await },
                    "random-access/parquet-tokio-local-disk".to_string(),
                    STORAGE_NVME.to_owned(),
                    &runtime,
                    &indices,
                    iterations,
                    target,
                )
            }
            #[cfg(feature = "lance")]
            Format::Lance => {
                let taxi_lance = runtime.block_on(taxi_data_lance())?;

                create_timing_measurement(
                    |indices| async { take_lance(&taxi_lance, indices).await },
                    "random-access/lance-tokio-local-disk".to_string(),
                    STORAGE_NVME.to_owned(),
                    &runtime,
                    &indices,
                    iterations,
                    target,
                )
            }
            Format::Csv | Format::Arrow | Format::OnDiskDuckDB => unimplemented!(),
        };

        targets.push(target);
        measurements.push(timing_measurement);

        progress.inc(1);
    }

    let mut writer: Box<dyn Write> = if let Some(output_path) = output_path {
        Box::new(File::create(output_path)?)
    } else {
        let stdout = stdout();
        Box::new(stdout.lock())
    };

    match display_format {
        DisplayFormat::Table => {
            render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements)?;
        }
    }

    progress.finish();
    Ok(())
}

fn validate_vortex_array(array: ArrayRef) {
    let struct_ = array.to_struct();
    assert_eq!(struct_.len(), 6, "expected 6 rows");
    let pu_location_id = struct_
        .column_by_name("PULocationID")
        .vortex_expect("could not get PULocationID");
    let do_location_id = struct_
        .column_by_name("DOLocationID")
        .vortex_expect("could not get DOLocationID");
    for (idx, loc) in [90i32, 249, 230, 79, 239, 236].iter().enumerate() {
        assert_eq!(
            pu_location_id.scalar_at(idx),
            Scalar::primitive(*loc, NonNullable)
        );
    }
    for (idx, loc) in [164i32, 231, 25, 224, 243, 239].iter().enumerate() {
        assert_eq!(
            do_location_id.scalar_at(idx),
            Scalar::primitive(*loc, NonNullable)
        );
    }
}
