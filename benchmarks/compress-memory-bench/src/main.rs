// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(feature = "dhat"))]
compile_error!("compress-memory-bench requires the `dhat` feature");

use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
#[cfg(feature = "lance")]
use compress_bench::LanceCompressor;
use compress_bench::parquet::ParquetCompressor;
use compress_bench::vortex::VortexCompressor;
use indicatif::ProgressBar;
use regex::Regex;
use vortex_bench::BenchmarkOutput;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Target;
use vortex_bench::compress::CompressOp;
use vortex_bench::compress::Compressor;
use vortex_bench::datasets::Dataset;
use vortex_bench::datasets::struct_list_of_ints::StructListOfInts;
use vortex_bench::datasets::taxi_data::TaxiData;
use vortex_bench::datasets::tpch_l_comment::TPCHLCommentCanonical;
use vortex_bench::datasets::tpch_l_comment::TPCHLCommentChunked;
use vortex_bench::dhat::start_heap_profiling;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::display::render_table;
use vortex_bench::downloadable_dataset::DownloadableDataset;
use vortex_bench::measurements::CompressionMemoryMeasurement;
use vortex_bench::public_bi::PBI_DATASETS;
use vortex_bench::public_bi::PBIDataset::Arade;
use vortex_bench::public_bi::PBIDataset::Bimbo;
use vortex_bench::public_bi::PBIDataset::CMSprovider;
use vortex_bench::public_bi::PBIDataset::Euro2016;
use vortex_bench::public_bi::PBIDataset::Food;
use vortex_bench::public_bi::PBIDataset::HashTags;
use vortex_bench::setup_logging_and_tracing;

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
    #[arg(
        long,
        value_enum,
        default_values_t = vec![CompressOp::Compress, CompressOp::Decompress]
    )]
    ops: Vec<CompressOp>,
    #[arg(long)]
    datasets: Option<String>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short, long)]
    output_path: Option<PathBuf>,
    #[arg(long)]
    tracing: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    run_compress_memory(
        args.datasets.map(|d| Regex::new(&d)).transpose()?,
        args.formats,
        args.ops,
        args.display_format,
        args.output_path,
    )
    .await
}

fn get_compressor(format: Format) -> Box<dyn Compressor> {
    match format {
        Format::OnDiskVortex => Box::new(VortexCompressor),
        Format::Parquet => Box::new(ParquetCompressor::new()),
        #[cfg(feature = "lance")]
        Format::Lance => Box::new(LanceCompressor),
        _ => unimplemented!("Compression memory benchmark not implemented for {format}"),
    }
}

fn measurement_name(op: CompressOp, bench_name: &str) -> String {
    match op {
        CompressOp::Compress => format!("compress peak memory/{bench_name}"),
        CompressOp::Decompress => format!("decompress peak memory/{bench_name}"),
    }
}

async fn measure_peak_memory(
    compressor: &dyn Compressor,
    parquet_path: &Path,
    op: CompressOp,
    bench_name: &str,
) -> anyhow::Result<CompressionMemoryMeasurement> {
    let profiler = start_heap_profiling()?;
    match op {
        CompressOp::Compress => {
            let _ = compressor.compress(parquet_path).await?;
        }
        CompressOp::Decompress => {
            let _ = compressor.decompress(parquet_path).await?;
        }
    }
    let stats = profiler.finish();

    Ok(CompressionMemoryMeasurement {
        name: measurement_name(op, bench_name),
        format: compressor.format(),
        value_mib: stats.max_mib(),
    })
}

async fn run_compress_memory(
    datasets_filter: Option<Regex>,
    formats: Vec<Format>,
    ops: Vec<CompressOp>,
    display_format: DisplayFormat,
    output_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    let targets = formats
        .iter()
        .map(|f| Target::new(Engine::default(), *f))
        .collect::<Vec<_>>();

    let struct_list_of_ints = [
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
        PBI_DATASETS.get(Euro2016),
        PBI_DATASETS.get(Food),
        PBI_DATASETS.get(HashTags),
        &TPCHLCommentChunked,
        &TPCHLCommentCanonical,
        &DownloadableDataset::RPlace,
        &DownloadableDataset::AirQuality,
    ]
    .into_iter()
    .chain(struct_list_of_ints.iter().map(|d| d as &dyn Dataset))
    .filter(|d| {
        if let Some(filter) = datasets_filter.as_ref() {
            filter.is_match(d.name())
        } else {
            d.name() != "airquality" && d.name() != "rplace"
        }
    })
    .collect();

    let progress = ProgressBar::new((datasets.len() * formats.len() * ops.len()) as u64);
    let mut measurements = Vec::new();

    for dataset in datasets {
        let parquet_path = dataset.to_parquet_path().await?;
        let bench_name = dataset.name();
        tracing::info!("Running memory benchmark for {bench_name}");

        for format in &formats {
            let compressor = get_compressor(*format);
            for op in &ops {
                measurements.push(
                    measure_peak_memory(compressor.as_ref(), &parquet_path, *op, bench_name)
                        .await?,
                );
                progress.inc(1);
            }
        }
    }

    progress.finish();

    let output = BenchmarkOutput::with_path("compress-memory", output_path);
    let mut writer = output.create_writer()?;

    match display_format {
        DisplayFormat::Table => render_table(&mut writer, measurements, &targets)?,
        DisplayFormat::GhJson => print_measurements_json(&mut writer, measurements)?,
    }

    Ok(())
}
