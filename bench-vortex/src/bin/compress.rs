// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::time::Duration;

use bench_vortex::compress::bench::{self as compress, CompressMeasurements, CompressOp};
use bench_vortex::datasets::Dataset;
use bench_vortex::datasets::struct_list_of_ints::StructListOfInts;
use bench_vortex::datasets::taxi_data::TaxiData;
use bench_vortex::datasets::tpch_l_comment::{TPCHLCommentCanonical, TPCHLCommentChunked};
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::downloadable_dataset::DownloadableDataset;
use bench_vortex::measurements::{CompressionTimingMeasurement, CustomUnitMeasurement};
use bench_vortex::public_bi::PBI_DATASETS;
use bench_vortex::public_bi::PBIDataset::{Arade, Bimbo, CMSprovider, Euro2016, Food, HashTags};
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{Engine, Format, Target, setup_logging_and_tracing};
use clap::Parser;
use indicatif::ProgressBar;
use itertools::Itertools;
use regex::Regex;
use tokio::runtime::Runtime;
use vortex::arrays::{ChunkedArray, ChunkedVTable};
use vortex::builders::builder_with_capacity;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::{Array, IntoArray};

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
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(
        long,
        value_enum,
        default_values_t = vec![CompressOp::Compress, CompressOp::Decompress]
    )]
    ops: Vec<CompressOp>,
    #[arg(long)]
    datasets: Option<String>,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short)]
    output_path: Option<PathBuf>,
    #[arg(long)]
    tracing: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let runtime = new_tokio_runtime(args.threads)?;

    compress(
        runtime,
        args.iterations,
        args.datasets.map(|d| Regex::new(&d)).transpose()?,
        args.formats,
        args.ops,
        args.display_format,
        &args.output_path,
    )
}

fn compress(
    runtime: Runtime,
    iterations: usize,
    datasets_filter: Option<Regex>,
    formats: Vec<Format>,
    ops: Vec<CompressOp>,
    display_format: DisplayFormat,
    output_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    let targets = formats
        .iter()
        .map(|f| Target::new(Engine::default(), *f))
        .collect_vec();

    let structlistofints = vec![
        StructListOfInts::new(100, 1000, 1),
        StructListOfInts::new(1000, 1000, 1),
        StructListOfInts::new(10000, 1000, 1),
        StructListOfInts::new(100, 1000, 50),
        StructListOfInts::new(1000, 1000, 50),
        StructListOfInts::new(10000, 1000, 50),
    ];

    let datasets: Vec<&dyn Dataset> = [
        &TaxiData as &dyn Dataset,
        PBI_DATASETS.get(Arade),
        PBI_DATASETS.get(Bimbo),
        PBI_DATASETS.get(CMSprovider),
        // Corporations, // duckdb thinks ' is a quote character but its used as an apostrophe
        // CityMaxCapita, // 11th column has F, M, and U but is inferred as boolean
        PBI_DATASETS.get(Euro2016),
        PBI_DATASETS.get(Food),
        PBI_DATASETS.get(HashTags),
        // Hatred, // panic in fsst_compress_iter
        // TableroSistemaPenal, // See bottom of this file
        // YaleLanguages, // 4th column looks like integer but also contains Y
        &TPCHLCommentChunked,
        &TPCHLCommentCanonical,
        &DownloadableDataset::RPlace,
        &DownloadableDataset::AirQuality,
    ]
    .into_iter()
    .chain(structlistofints.iter().map(|d| d as &dyn Dataset))
    .filter(|d| {
        if let Some(filter) = datasets_filter.as_ref() {
            filter.is_match(d.name())
        } else {
            // These download data from pcodec's public bucket, presumably creating egress charges
            // for pcodec. As such, we do not run in CI.
            d.name() != "airquality" && d.name() != "rplace"
        }
    })
    .collect();

    let progress = ProgressBar::new((datasets.len() * formats.len() * ops.len()) as u64);

    let measurements = datasets
        .into_iter()
        .map(|dataset_handle| {
            benchmark_compress(
                &runtime,
                &progress,
                &formats,
                &ops,
                iterations,
                dataset_handle,
            )
        })
        .try_collect::<_, Vec<_>, _>()?
        .into_iter()
        .collect::<CompressMeasurements>();

    progress.finish();

    let mut writer: Box<dyn Write> = if let Some(output_path) = output_path {
        Box::new(File::create(output_path)?)
    } else {
        let stdout = stdout();
        Box::new(stdout.lock())
    };

    match display_format {
        DisplayFormat::Table => {
            render_table(&mut writer, measurements.timings, &targets)?;
            render_table(
                &mut writer,
                measurements.ratios,
                &if formats.contains(&Format::OnDiskVortex) {
                    vec![Target::new(Engine::default(), Format::OnDiskVortex)]
                } else {
                    vec![]
                },
            )
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements.timings)?;
            print_measurements_json(&mut writer, measurements.ratios)
        }
    }
}

// Type aliases for compression and decompression function signatures.
type CompressFn = fn(
    &Runtime,
    &dyn Array,
    usize,
    &str,
) -> anyhow::Result<(
    Duration,
    u64,
    Vec<CustomUnitMeasurement>,
    CompressionTimingMeasurement,
)>;

type DecompressFn = fn(
    &Runtime,
    &dyn Array,
    usize,
    &str,
) -> anyhow::Result<(Duration, CompressionTimingMeasurement)>;

pub fn benchmark_compress(
    runtime: &Runtime,
    progress: &ProgressBar,
    formats: &[Format],
    ops: &[CompressOp],
    iterations: usize,
    dataset_handle: &dyn Dataset,
) -> anyhow::Result<CompressMeasurements> {
    let bench_name = dataset_handle.name();
    tracing::info!("Running {bench_name} benchmark");

    let vx_array = runtime.block_on(async { dataset_handle.to_vortex_array().await })?;
    let uncompressed = ChunkedArray::from_iter(
        vx_array
            .as_::<ChunkedVTable>()
            .chunks()
            .iter()
            .map(|chunk| {
                let mut builder = builder_with_capacity(chunk.dtype(), chunk.len());
                chunk.append_to_builder(builder.as_mut());
                builder.finish()
            }),
    )
    .into_array();

    let mut ratios = Vec::new();
    let mut timings = Vec::new();
    let mut measurements_map: HashMap<(Format, CompressOp), Duration> = HashMap::new();
    let mut compressed_sizes: HashMap<Format, u64> = HashMap::new();

    for format in formats {
        for op in ops {
            let result = match op {
                CompressOp::Compress => {
                    // Select the compression function based on format.
                    let compress_fn: Option<CompressFn> = match format {
                        Format::OnDiskVortex => Some(compress::benchmark_vortex_compress),
                        Format::Parquet => Some(compress::benchmark_parquet_compress),
                        // Format::Lance => Some(compress::benchmark_lance_compress),
                        _ => None,
                    };

                    if let Some(func) = compress_fn {
                        let (time, size, ratios_part, timing) =
                            func(runtime, &uncompressed, iterations, bench_name)?;
                        compressed_sizes.insert(*format, size);
                        ratios.extend(ratios_part);
                        timings.push(timing);
                        Some(time)
                    } else {
                        eprintln!("{op} benchmark on {format} not supported");
                        None
                    }
                }
                CompressOp::Decompress => {
                    // Select the decompression function based on format.
                    let decompress_fn: Option<DecompressFn> = match format {
                        Format::OnDiskVortex => Some(compress::benchmark_vortex_decompress),
                        Format::Parquet => Some(compress::benchmark_parquet_decompress),
                        // Format::Lance => Some(compress::benchmark_lance_decompress),
                        _ => None,
                    };

                    if let Some(func) = decompress_fn {
                        let (time, timing) = func(runtime, &uncompressed, iterations, bench_name)?;
                        timings.push(timing);
                        Some(time)
                    } else {
                        eprintln!("{op} benchmark on {format} not supported");
                        None
                    }
                }
            };

            if let Some(time) = result {
                measurements_map.insert((*format, *op), time);
                progress.inc(1);
            }
        }
    }

    // Calculate cross-format ratios after all measurements
    compress::calculate_ratios(
        &measurements_map,
        &compressed_sizes,
        bench_name,
        &mut ratios,
    );

    Ok(CompressMeasurements { timings, ratios })
}

/*

For the TableroSistemaPenal dataset, we get this error:

thread 'main' panicked at bench-vortex/benches/compress_benchmark.rs:224:42: called `Result::unwrap()` on an `Err` value: expected type: {column00=utf8?, column01=i64?, column02=utf8?, column03=f64?, column04=i64?, column05=utf8?, column06=utf8?, column07=utf8?, column08=utf8?, column09=utf8?, column10=i64?, column11=i64?, column12=utf8?, column13=utf8?, column14=i64?, column15=i64?, column16=utf8?, column17=utf8?, column18=utf8?, column19=utf8?, column20=i64?, column21=utf8?, column22=utf8?, column23=utf8?, column24=utf8?, column25=i64?, column26=utf8?} but instead got {column00=utf8?, column01=i64?, column02=i64?, column03=i64?, column04=i64?, column05=utf8?, column06=i64?, column07=i64?, column08=i64?, column09=utf8?, column10=ext(vortex.date, ExtMetadata([4]))?, column11=ext(vortex.date, ExtMetadata([4]))?, column12=utf8?, column13=utf8?, column14=utf8?, column15=i64?, column16=i64?, column17=utf8?, column18=utf8?, column19=utf8?, column20=utf8?, column21=utf8?}

*/
