// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
#[cfg(feature = "lance")]
use compress_bench::LanceCompressor;
use compress_bench::parquet::ParquetCompressor;
use compress_bench::vortex::VortexCompressor;
use indicatif::ProgressBar;
use itertools::Itertools;
use regex::Regex;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::LogFormat;
use vortex_bench::Target;
use vortex_bench::compress::CompressMeasurements;
use vortex_bench::compress::CompressOp;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::READ_PROJECTION_COLUMNS;
use vortex_bench::compress::READ_PROJECTION_ROOT_COLUMNS;
use vortex_bench::compress::benchmark_compress;
use vortex_bench::compress::benchmark_decompress;
use vortex_bench::compress::calculate_ratios;
use vortex_bench::create_output_writer;
use vortex_bench::datasets::Dataset;
use vortex_bench::datasets::struct_list_of_ints::StructListOfInts;
use vortex_bench::datasets::taxi_data::TaxiData;
use vortex_bench::datasets::tpch_l_comment::TPCHLCommentCanonical;
use vortex_bench::datasets::tpch_l_comment::TPCHLCommentChunked;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::display::render_table;
use vortex_bench::downloadable_dataset::DownloadableDataset;
use vortex_bench::public_bi::PBI_DATASETS;
use vortex_bench::public_bi::PBIDataset::Arade;
use vortex_bench::public_bi::PBIDataset::Bimbo;
use vortex_bench::public_bi::PBIDataset::CMSprovider;
use vortex_bench::public_bi::PBIDataset::Euro2016;
use vortex_bench::public_bi::PBIDataset::Food;
use vortex_bench::public_bi::PBIDataset::HashTags;
use vortex_bench::setup_logging_and_tracing_with_format;
use vortex_bench::v3;

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
    #[arg(short, long)]
    output_path: Option<PathBuf>,
    /// Additionally write v3 JSONL records to this path. See
    /// `benchmarks-website/planning/02-contracts.md`.
    #[arg(long)]
    gh_json_v3: Option<PathBuf>,
    #[arg(long)]
    tracing: bool,
    /// Format for the primary stderr log sink. `text` is the default human-readable format;
    /// `json` emits one JSON object per event, suitable for piping into `jq`.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing_with_format(args.verbose, args.tracing, args.log_format)?;

    run_compress(
        args.iterations,
        args.datasets.map(|d| Regex::new(&d)).transpose()?,
        args.formats,
        args.ops,
        args.display_format,
        args.output_path,
        args.gh_json_v3,
    )
    .await
}

/// Get a compressor for the given format.
fn get_compressor(format: Format) -> Box<dyn Compressor> {
    match format {
        Format::OnDiskVortex => Box::new(VortexCompressor),
        Format::Parquet => Box::new(ParquetCompressor::new()),
        #[cfg(feature = "lance")]
        Format::Lance => Box::new(LanceCompressor),
        _ => unimplemented!("Compress bench not implemented for {format}"),
    }
}

/// The benchmark ID used for output path.
const BENCHMARK_ID: &str = "compress";

async fn run_compress(
    iterations: usize,
    datasets_filter: Option<Regex>,
    formats: Vec<Format>,
    ops: Vec<CompressOp>,
    display_format: DisplayFormat,
    output_path: Option<PathBuf>,
    gh_json_v3: Option<PathBuf>,
) -> anyhow::Result<()> {
    let targets = formats
        .iter()
        .map(|f| Target::new(Engine::default(), *f))
        .collect_vec();

    let structlistofints = [
        StructListOfInts::new(100, 1000, 1),
        StructListOfInts::new(1000, 1000, 1),
        StructListOfInts::new(10000, 1000, 1),
        StructListOfInts::new(100, 1000, 50),
        StructListOfInts::new(1000, 1000, 50),
        StructListOfInts::new(10000, 1000, 50),
        // Very wide file: project a fixed 10k columns out of 100k, across 10 chunks.
        StructListOfInts::new_with_projection(
            READ_PROJECTION_ROOT_COLUMNS,
            1000,
            10,
            Some(READ_PROJECTION_COLUMNS),
        ),
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

    let mut measurements = vec![];
    let mut v3_records: Vec<v3::V3Record> = Vec::new();

    for dataset_handle in datasets.into_iter() {
        let (m, mut records) =
            run_benchmark_for_dataset(&progress, &formats, &ops, iterations, dataset_handle)
                .await?;
        measurements.push(m);
        v3_records.append(&mut records);
    }

    let measurements = CompressMeasurements::from_iter(measurements);

    progress.finish();

    if let Some(path) = gh_json_v3 {
        v3::write_jsonl_to_path(&path, &v3_records)?;
    }

    let mut writer = create_output_writer(&display_format, output_path, BENCHMARK_ID)?;

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

async fn run_benchmark_for_dataset(
    progress: &ProgressBar,
    formats: &[Format],
    ops: &[CompressOp],
    iterations: usize,
    dataset_handle: &dyn Dataset,
) -> anyhow::Result<(CompressMeasurements, Vec<v3::V3Record>)> {
    let bench_name = dataset_handle.name();
    let (v3_dataset, v3_variant) = dataset_handle.v3_dataset_dims();
    tracing::info!("Running {bench_name} benchmark");

    // Get the parquet file path for this dataset
    let parquet_path = dataset_handle.to_parquet_path().await?;

    let mut ratios = Vec::new();
    let mut timings = Vec::new();
    let mut measurements_map: HashMap<(Format, CompressOp), Duration> = HashMap::new();
    let mut compressed_sizes: HashMap<Format, u64> = HashMap::new();
    let mut v3_records: Vec<v3::V3Record> = Vec::new();

    for format in formats {
        let compressor = get_compressor(*format);

        for op in ops {
            let time = match op {
                CompressOp::Compress => {
                    let result = benchmark_compress(
                        compressor.as_ref(),
                        &parquet_path,
                        iterations,
                        bench_name,
                    )
                    .await?;
                    compressed_sizes.insert(*format, result.compressed_size);
                    let all_runs_ns: Vec<u64> = result
                        .all_runs
                        .iter()
                        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
                        .collect();
                    v3_records.push(v3::compression_time_record(
                        &result.timing,
                        v3_dataset,
                        v3_variant,
                        CompressOp::Compress,
                        all_runs_ns,
                    ));
                    v3_records.push(v3::compression_size_record(
                        v3_dataset,
                        v3_variant,
                        *format,
                        result.compressed_size,
                    ));
                    ratios.extend(result.ratios);
                    timings.push(result.timing);
                    result.time
                }
                CompressOp::Decompress => {
                    let result = benchmark_decompress(
                        compressor.as_ref(),
                        &parquet_path,
                        iterations,
                        bench_name,
                    )
                    .await?;
                    let all_runs_ns: Vec<u64> = result
                        .all_runs
                        .iter()
                        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
                        .collect();
                    v3_records.push(v3::compression_time_record(
                        &result.timing,
                        v3_dataset,
                        v3_variant,
                        CompressOp::Decompress,
                        all_runs_ns,
                    ));
                    timings.push(result.timing);
                    result.time
                }
            };

            measurements_map.insert((*format, *op), time);
            progress.inc(1);
        }
    }

    // Calculate cross-format ratios after all measurements.
    calculate_ratios(
        &measurements_map,
        &compressed_sizes,
        bench_name,
        &mut ratios,
    );

    Ok((CompressMeasurements { timings, ratios }, v3_records))
}
