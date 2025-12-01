// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Write;
use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;

use bench_vortex::Engine;
use bench_vortex::Format;
use bench_vortex::Target;
use bench_vortex::compress::bench as compress;
use bench_vortex::compress::bench::CompressMeasurements;
use bench_vortex::compress::bench::CompressOp;
use bench_vortex::datasets::Dataset;
use bench_vortex::datasets::struct_list_of_ints::StructListOfInts;
use bench_vortex::datasets::taxi_data::TaxiData;
use bench_vortex::datasets::tpch_l_comment::TPCHLCommentCanonical;
use bench_vortex::datasets::tpch_l_comment::TPCHLCommentChunked;
use bench_vortex::display::DisplayFormat;
use bench_vortex::display::print_measurements_json;
use bench_vortex::display::render_table;
use bench_vortex::downloadable_dataset::DownloadableDataset;
use bench_vortex::measurements::CompressionTimingMeasurement;
use bench_vortex::measurements::CustomUnitMeasurement;
use bench_vortex::public_bi::PBI_DATASETS;
use bench_vortex::public_bi::PBIDataset::Arade;
use bench_vortex::public_bi::PBIDataset::Bimbo;
use bench_vortex::public_bi::PBIDataset::CMSprovider;
use bench_vortex::public_bi::PBIDataset::Euro2016;
use bench_vortex::public_bi::PBIDataset::Food;
use bench_vortex::public_bi::PBIDataset::HashTags;
use bench_vortex::setup_logging_and_tracing;
use bench_vortex::utils::new_tokio_runtime;
use clap::Parser;
use indicatif::ProgressBar;
use itertools::Itertools;
use regex::Regex;
use tokio::runtime::Runtime;
use vortex::array::Array;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ChunkedVTable;
use vortex::array::builders::builder_with_capacity;
use vortex::utils::aliases::hash_map::HashMap;

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
        // TableroSistemaPenal, // Unexpected type error
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
            let time = match op {
                CompressOp::Compress => {
                    // Select the compression function based on format.
                    let compress_fn: CompressFn = match format {
                        Format::OnDiskVortex => compress::benchmark_vortex_compress,
                        Format::Parquet => compress::benchmark_parquet_compress,
                        #[cfg(feature = "lance")]
                        Format::Lance => compress::benchmark_lance_compress,
                        _ => unimplemented!("Compress bench not implemented for {format}"),
                    };

                    let (time, size, ratios_part, timing) =
                        compress_fn(runtime, &uncompressed, iterations, bench_name)?;
                    compressed_sizes.insert(*format, size);
                    ratios.extend(ratios_part);
                    timings.push(timing);

                    time
                }
                CompressOp::Decompress => {
                    // Select the decompression function based on format.
                    let decompress_fn: DecompressFn = match format {
                        Format::OnDiskVortex => compress::benchmark_vortex_decompress,
                        Format::Parquet => compress::benchmark_parquet_decompress,
                        #[cfg(feature = "lance")]
                        Format::Lance => compress::benchmark_lance_decompress,
                        _ => unimplemented!("Decompress bench not implemented for {format}"),
                    };

                    let (time, timing) =
                        decompress_fn(runtime, &uncompressed, iterations, bench_name)?;
                    timings.push(timing);
                    time
                }
            };

            measurements_map.insert((*format, *op), time);
            progress.inc(1);
        }
    }

    // Calculate cross-format ratios after all measurements.
    compress::calculate_ratios(
        &measurements_map,
        &compressed_sizes,
        bench_name,
        &mut ratios,
    );

    Ok(CompressMeasurements { timings, ratios })
}
