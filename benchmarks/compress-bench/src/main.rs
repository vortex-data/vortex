// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
#[cfg(feature = "lance")]
use compress_bench::LanceCompressor;
#[cfg(feature = "cuda")]
use compress_bench::gpu_vortex::GpuVortexCompressor;
use compress_bench::parquet::ParquetCompressor;
use compress_bench::vortex::VortexCompressor;
use indicatif::ProgressBar;
use itertools::Itertools;
use regex::Regex;
use tokio::sync::Semaphore;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::parallelism::get_available_parallelism;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::LogFormat;
use vortex_bench::Target;
use vortex_bench::compress::CompressMeasurements;
use vortex_bench::compress::CompressOp;
use vortex_bench::compress::Compressor;
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
    /// Run GPU decompression for the allow-listed benchmarks.
    ///
    /// This filters the suite to GPU-supported dataset names and runs only Vortex decompression.
    #[arg(long)]
    gpu_decompress: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short, long)]
    output_path: Option<PathBuf>,
    /// Additionally write benchmark ingest JSONL records to this path.
    #[arg(long = "ingest-jsonl")]
    ingest_output: Option<PathBuf>,
    #[arg(long)]
    tracing: bool,
    /// Materialize every dataset's data, report the setup timing, and exit without
    /// benchmarking. Use on a cold cache to measure download and conversion cost.
    #[arg(long)]
    setup_only: bool,
    /// Format for the primary stderr log sink. `text` is the default human-readable format;
    /// `json` emits one JSON object per event, suitable for piping into `jq`.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing_with_format(args.verbose, args.tracing, args.log_format)?;

    if args.gpu_decompress && !cfg!(feature = "cuda") {
        anyhow::bail!("--gpu-decompress requires building compress-bench with --features cuda");
    }

    let (formats, ops) = if args.gpu_decompress {
        (vec![Format::OnDiskVortex], vec![CompressOp::Decompress])
    } else {
        (args.formats, args.ops)
    };

    run_compress(
        args.iterations,
        args.datasets.map(|d| Regex::new(&d)).transpose()?,
        formats,
        ops,
        args.gpu_decompress,
        args.display_format,
        args.output_path,
        args.ingest_output,
        args.setup_only,
    )
    .await
}

/// Get a compressor for the given format.
fn get_compressor(format: Format, gpu_decompress: bool) -> Box<dyn Compressor> {
    if gpu_decompress {
        #[cfg(feature = "cuda")]
        {
            return Box::new(GpuVortexCompressor);
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!("GPU feature validation happens before selecting compressors");
    }

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

/// Repo-relative path of the suite explainer linked from CI benchmark PR comments.
const DOC_PATH: &str = "benchmarks/compress-bench/README.md";

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark CLI options are forwarded one-to-one"
)]
async fn run_compress(
    iterations: usize,
    datasets_filter: Option<Regex>,
    formats: Vec<Format>,
    ops: Vec<CompressOp>,
    gpu_decompress: bool,
    display_format: DisplayFormat,
    output_path: Option<PathBuf>,
    ingest_output: Option<PathBuf>,
    setup_only: bool,
) -> anyhow::Result<()> {
    let targets = formats
        .iter()
        .map(|f| Target::new(Engine::default(), *f))
        .collect_vec();

    // Leaked so the setup phase can `tokio::spawn` per dataset, which needs `'static`. The
    // other datasets are already static (unit structs and a `LazyLock` registry).
    let structlistofints: &'static [StructListOfInts; 6] = Box::leak(Box::new([
        StructListOfInts::new(100, 1000, 1),
        StructListOfInts::new(1000, 1000, 1),
        StructListOfInts::new(10000, 1000, 1),
        StructListOfInts::new(100, 1000, 50),
        StructListOfInts::new(1000, 1000, 50),
        StructListOfInts::new(10000, 1000, 50),
        // See https://github.com/vortex-data/vortex/issues/8330
        // Very wide file: project a fixed 10k columns out of 100k, across 10 chunks.
        // StructListOfInts::new_with_projection(
        //     READ_PROJECTION_ROOT_COLUMNS,
        //     1000,
        //     10,
        //     Some(READ_PROJECTION_COLUMNS),
        // ),
    ]));

    // Add an existing benchmark name here only after its CUDA-compatible compression and
    // decompression kernels have been verified end to end.
    #[expect(
        clippy::useless_vec,
        reason = "this is an intentionally incremental allow-list of benchmark names"
    )]
    let gpu_decompress_benchmarks = vec!["TPC-H l_comment canonical"];

    let datasets: Vec<&'static dyn Dataset> = [
        &TaxiData as &'static dyn Dataset,
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
    .chain(structlistofints.iter().map(|d| d as &'static dyn Dataset))
    .filter(|d| {
        if gpu_decompress && !gpu_decompress_benchmarks.contains(&d.name()) {
            return false;
        }
        if let Some(filter) = datasets_filter.as_ref() {
            filter.is_match(d.name())
        } else {
            // These download data from pcodec's public bucket, presumably creating egress charges
            // for pcodec. As such, we do not run in CI.
            d.name() != "airquality" && d.name() != "rplace"
        }
    })
    .collect();

    let setup = prepare_datasets(&datasets).await?;
    setup.report();
    if setup_only {
        return Ok(());
    }

    let progress = ProgressBar::new((datasets.len() * formats.len() * ops.len()) as u64);

    let mut measurements = vec![];
    let mut v3_records: Vec<v3::V3Record> = Vec::new();

    for dataset_handle in datasets.into_iter() {
        let (m, mut records) = run_benchmark_for_dataset(
            &progress,
            &formats,
            &ops,
            iterations,
            dataset_handle,
            gpu_decompress,
        )
        .await?;
        measurements.push(m);
        v3_records.append(&mut records);
    }

    let measurements = CompressMeasurements::from_iter(measurements);

    progress.finish();

    if let Some(path) = ingest_output {
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
            print_measurements_json(&mut writer, measurements.timings, DOC_PATH)?;
            print_measurements_json(&mut writer, measurements.ratios, DOC_PATH)
        }
    }
}

/// Wall-clock cost of materializing every dataset, split by stage.
struct SetupTiming {
    download: Duration,
    convert: Duration,
    /// Per-dataset convert time, slowest first.
    per_dataset: Vec<(String, Duration)>,
    concurrency: usize,
}

impl SetupTiming {
    fn total(&self) -> Duration {
        self.download + self.convert
    }

    /// Log the phase breakdown. Setup is idempotent, so this is only meaningful on a cold
    /// cache — a warm run reports near-zero and says nothing about the parallelism.
    fn report(&self) {
        tracing::info!(
            "setup: {:.1}s total ({:.1}s download, {:.1}s convert at concurrency {})",
            self.total().as_secs_f64(),
            self.download.as_secs_f64(),
            self.convert.as_secs_f64(),
            self.concurrency,
        );
        let serial: Duration = self.per_dataset.iter().map(|(_, d)| *d).sum();
        tracing::info!(
            "setup: convert was {:.1}s of work across {} datasets, {:.1}x speedup from overlap",
            serial.as_secs_f64(),
            self.per_dataset.len(),
            serial.as_secs_f64() / self.convert.as_secs_f64().max(f64::MIN_POSITIVE),
        );
        for (name, elapsed) in self.per_dataset.iter().take(5) {
            tracing::info!("setup:   {name}: {:.1}s", elapsed.as_secs_f64());
        }
    }
}

/// Materialize every dataset's Parquet before any benchmark runs, in two stages.
///
/// Downloads go first and all at once: they are network bound, and the shared pool caps
/// total in-flight requests. Conversion (bz2 decompress, CSV to Parquet, generation) is CPU
/// and disk bound, so it is capped at the available parallelism instead — which under
/// `scripts/bench-taskset.sh` is the pinned NUMA node's core count, not the whole machine.
async fn prepare_datasets(datasets: &[&'static dyn Dataset]) -> anyhow::Result<SetupTiming> {
    let started = Instant::now();
    futures::future::try_join_all(datasets.iter().map(|d| d.download())).await?;
    let download = started.elapsed();

    let concurrency = get_available_parallelism().unwrap_or(1).max(1);
    let permits = Arc::new(Semaphore::new(concurrency));
    let started = Instant::now();

    // `tokio::spawn`, not `buffer_unordered`: `to_parquet_path` does its generation and
    // conversion work synchronously inside an async fn, so it never yields. Polled from a
    // single task the futures would run strictly one after another — spawning puts each on
    // its own runtime worker, and the semaphore keeps that bounded. This mirrors
    // `convert_parquet_directory_to_vortex`.
    let mut per_dataset: Vec<(String, Duration)> =
        futures::future::try_join_all(datasets.iter().map(|d| {
            let permits = Arc::clone(&permits);
            let d = *d;
            tokio::spawn(async move {
                let _permit = permits.acquire().await?;
                let started = Instant::now();
                d.to_parquet_path().await?;
                anyhow::Ok((d.name().to_owned(), started.elapsed()))
            })
        }))
        .await?
        .into_iter()
        .collect::<anyhow::Result<Vec<_>>>()?;
    let convert = started.elapsed();

    per_dataset.sort_by_key(|(_, elapsed)| std::cmp::Reverse(*elapsed));
    Ok(SetupTiming {
        download,
        convert,
        per_dataset,
        concurrency,
    })
}

async fn run_benchmark_for_dataset(
    progress: &ProgressBar,
    formats: &[Format],
    ops: &[CompressOp],
    iterations: usize,
    dataset_handle: &dyn Dataset,
    gpu_decompress: bool,
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
        let compressor = get_compressor(*format, gpu_decompress);

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
                        if gpu_decompress {
                            Some("gpu")
                        } else {
                            v3_variant
                        },
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
