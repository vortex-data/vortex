// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
#[cfg(feature = "lance")]
use compress_bench::LanceCompressor;
use compress_bench::gpu::GpuCodec;
use compress_bench::gpu::GpuOptions;
use compress_bench::gpu::compressor as gpu_compressor;
use compress_bench::parquet::ParquetCompressor;
use compress_bench::vortex::VortexCompressor;
use futures::FutureExt;
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
use vortex_bench::measurements::CustomUnitMeasurement;
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
    /// Print the dataset names that would run, one per line, and exit.
    /// Knowledge of datasets lies only in this binary so we need
    /// orchestrator to know whan queries to run one by one.
    #[arg(long, default_value_t = false)]
    print_datasets: bool,
    /// Run GPU decompression for the GPU-supported benchmarks.
    ///
    /// Restricts the suite to datasets with verified CUDA decode support and measures
    /// decompression only, for both Vortex and Parquet.
    #[arg(long)]
    gpu_decompress: bool,
    /// Page codec the GPU Parquet file is written with.
    ///
    /// Snappy is the Parquet default and the codec GPU readers decompress fastest.
    #[arg(long, value_enum, default_value_t)]
    gpu_parquet_codec: GpuCodec,
    /// Cross-check every GPU-decompressed result against the CPU decoder.
    ///
    /// The check runs before the timed measurement and is not included in it, so the numbers a
    /// verifying run publishes are still comparable — it only makes the run slower.
    #[arg(long)]
    gpu_verify: bool,
    /// Read the Vortex GPU file with direct IO, bypassing the page cache.
    ///
    /// Off by default: cuDF reads through the page cache after an untimed warm-up, so direct
    /// IO would compare a Vortex read of the disk against a cuDF read of RAM. Turn it on to
    /// measure storage bandwidth instead, and do not read the ratio as a decode comparison.
    #[arg(long)]
    gpu_direct_io: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short, long)]
    output_path: Option<PathBuf>,
    /// Additionally write benchmark ingest JSONL records to this path.
    #[arg(long = "ingest-jsonl")]
    ingest_output: Option<PathBuf>,
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

    if args.gpu_decompress && !cfg!(feature = "cuda") {
        anyhow::bail!("--gpu-decompress requires building compress-bench with --features cuda");
    }

    let mode = if args.gpu_decompress {
        BenchMode::Gpu(GpuOptions {
            codec: args.gpu_parquet_codec,
            verify: args.gpu_verify,
            direct_io: args.gpu_direct_io,
        })
    } else {
        BenchMode::Cpu
    };

    let (formats, ops) = if mode.is_gpu() {
        (
            vec![Format::Parquet, Format::OnDiskVortex],
            vec![CompressOp::Decompress],
        )
    } else {
        (args.formats, args.ops)
    };

    run_compress(
        args.iterations,
        args.datasets.map(|d| Regex::new(&d)).transpose()?,
        formats,
        ops,
        mode,
        args.print_datasets,
        args.display_format,
        args.output_path,
        args.ingest_output,
    )
    .await
}

/// Which suite a run measures.
///
/// GPU mode is not a setting on the CPU suite: it selects a different dataset list, restricts the
/// ops to decompression, swaps both compressors and labels its ratio differently. A variant says
/// that; an `Option<GpuOptions>` only said "maybe some GPU settings", leaving "which suite is
/// this?" and "how is the GPU configured?" to be the same question.
#[derive(Clone, Copy, Debug)]
enum BenchMode {
    /// The ordinary host suite.
    Cpu,
    /// Device decompression, configured by the `--gpu-*` flags.
    Gpu(GpuOptions),
}

impl BenchMode {
    /// Whether this run measures the GPU.
    fn is_gpu(self) -> bool {
        matches!(self, BenchMode::Gpu(_))
    }
}

/// Get a compressor for the given format.
fn get_compressor(format: Format, mode: BenchMode) -> Box<dyn Compressor> {
    if let BenchMode::Gpu(options) = mode {
        return gpu_compressor(format, options);
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
    mode: BenchMode,
    print_datasets: bool,
    display_format: DisplayFormat,
    output_path: Option<PathBuf>,
    ingest_output: Option<PathBuf>,
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
        // See https://github.com/vortex-data/vortex/issues/8330
        // Very wide file: project a fixed 10k columns out of 100k, across 10 chunks.
        // StructListOfInts::new_with_projection(
        //     READ_PROJECTION_ROOT_COLUMNS,
        //     1000,
        //     10,
        //     Some(READ_PROJECTION_COLUMNS),
        // ),
    ];

    // Datasets run in GPU mode. Add one only after a `--gpu-verify` run has confirmed its CUDA
    // decode end to end; a dataset here that cannot decode fails the benchmark job. Between them
    // these cover FSST strings, bit-packed numerics and columns with nulls.
    //
    // Not yet listed, each blocked on a `vortex-cuda` gap rather than on the benchmark:
    //
    // - `taxi` and `Arade`: `Unsupported ptype u16`. The CUDA `date_time_parts` kernel dispatches
    //   with `match_each_signed_integer_ptype!` where the CPU canonicaliser uses
    //   `match_each_integer_ptype!`. Widening the fused kernel takes 4³ = 64 PTX instantiations
    //   to 8³ = 512, so it is not a free change.
    // - `Euro2016` and `HashTags`: `No CUDA kernel for encoding vortex.masked`.
    // - `CMSprovider`: `expected host buffer` — a CPU fallback is reached with device-resident
    //   buffers, which `CudaArrayExt::execute_cuda` refuses.
    // - `StructListOfInts`: its list layouts have no verified CUDA decode path.
    let gpu_datasets: [&dyn Dataset; 4] = [
        &TPCHLCommentCanonical as &dyn Dataset,
        &TPCHLCommentChunked,
        PBI_DATASETS.get(Bimbo),
        PBI_DATASETS.get(Food),
    ];

    let all_datasets: Vec<&dyn Dataset> = [
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
    .collect();

    let datasets: Vec<&dyn Dataset> = if mode.is_gpu() {
        gpu_datasets.to_vec()
    } else {
        all_datasets
    }
    .into_iter()
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

    if print_datasets {
        for dataset in &datasets {
            println!("{}", dataset.name());
        }
        return Ok(());
    }

    let progress = ProgressBar::new((datasets.len() * formats.len() * ops.len()) as u64);

    let mut measurements = vec![];
    let mut v3_records: Vec<v3::V3Record> = Vec::new();

    // A GPU pass reports on every dataset rather than stopping at the first failure, so one run
    // says which datasets decode correctly on the GPU and still yields numbers for the rest.
    let survey_all = mode.is_gpu();
    let mut failures: Vec<(String, anyhow::Error)> = Vec::new();

    for dataset_handle in datasets.into_iter() {
        let run =
            run_benchmark_for_dataset(&progress, &formats, &ops, iterations, dataset_handle, mode);

        // Missing CUDA kernel support surfaces as a panic rather than an error, so the survey
        // has to catch those too or the first unsupported dataset ends the run.
        let result = if survey_all {
            match AssertUnwindSafe(run).catch_unwind().await {
                Ok(result) => result,
                Err(panic) => Err(anyhow::anyhow!("panicked: {}", panic_message(&panic))),
            }
        } else {
            run.await
        };

        match result {
            Ok((m, mut records)) => {
                measurements.push(m);
                v3_records.append(&mut records);
            }
            Err(error) if survey_all => {
                tracing::error!("{}: {error:#}", dataset_handle.name());
                failures.push((dataset_handle.name().to_string(), error));
            }
            Err(error) => return Err(error),
        }
    }

    let measurements = CompressMeasurements::from_iter(measurements);

    progress.finish();

    if let Some(path) = ingest_output {
        v3::write_jsonl_to_path(&path, &v3_records)?;
    }

    let mut writer = create_output_writer(&display_format, output_path, BENCHMARK_ID)?;

    // The tables render before any failure is reported, so a partially failing GPU matrix still
    // publishes the numbers for the datasets that did decode.
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
            )?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements.timings, DOC_PATH)?;
            print_measurements_json(&mut writer, measurements.ratios, DOC_PATH)?;
        }
    }

    if !failures.is_empty() {
        eprintln!(
            "\nGPU decompression failed for {} dataset(s):",
            failures.len()
        );
        for (dataset, error) in &failures {
            eprintln!("  - {dataset}: {error:#}");
        }
        anyhow::bail!(
            "GPU decompression failed for: {}",
            failures
                .iter()
                .map(|(dataset, _)| dataset.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    Ok(())
}

async fn run_benchmark_for_dataset(
    progress: &ProgressBar,
    formats: &[Format],
    ops: &[CompressOp],
    iterations: usize,
    dataset_handle: &dyn Dataset,
    mode: BenchMode,
) -> anyhow::Result<(CompressMeasurements, Vec<v3::V3Record>)> {
    let bench_name = dataset_handle.name();
    // A GPU decode and a host decode of the same dataset would otherwise publish the same
    // measurement name, and the PR benchmark report matches rows by name — so a GPU run is
    // diffed against whatever the host suite last recorded and every dataset reads as a
    // several-fold regression. The ratio rows already say `gpu`; the timings did not.
    let decompress_name = if matches!(mode, BenchMode::Gpu(_)) {
        format!("{bench_name} gpu")
    } else {
        bench_name.to_string()
    };
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
        let compressor = get_compressor(*format, mode);

        for op in ops {
            let time = match op {
                CompressOp::Compress => {
                    let result = benchmark_compress(
                        compressor.as_ref(),
                        &parquet_path,
                        iterations,
                        bench_name,
                    )
                    .await
                    .with_context(|| format!("compressing {bench_name} as {format}"))?;
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
                        &decompress_name,
                    )
                    .await
                    .with_context(|| format!("decompressing {bench_name} as {format}"))?;
                    let all_runs_ns: Vec<u64> = result
                        .all_runs
                        .iter()
                        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
                        .collect();
                    v3_records.push(v3::compression_time_record(
                        &result.timing,
                        v3_dataset,
                        if mode.is_gpu() {
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
    match mode {
        // The shared ratio labels name the CPU suite's codec, which the GPU run does not
        // necessarily use, so GPU mode emits its own correctly-labelled ratio.
        BenchMode::Gpu(options) => {
            push_gpu_ratio(&measurements_map, options, bench_name, &mut ratios)
        }
        BenchMode::Cpu => calculate_ratios(
            &measurements_map,
            &compressed_sizes,
            bench_name,
            &mut ratios,
        ),
    }

    Ok((CompressMeasurements { timings, ratios }, v3_records))
}

/// Emit the Vortex-versus-Parquet decompression ratio for a GPU run.
fn push_gpu_ratio(
    measurements: &HashMap<(Format, CompressOp), Duration>,
    options: GpuOptions,
    bench_name: &str,
    ratios: &mut Vec<CustomUnitMeasurement>,
) {
    let (Some(vortex_time), Some(parquet_time)) = (
        measurements.get(&(Format::OnDiskVortex, CompressOp::Decompress)),
        measurements.get(&(Format::Parquet, CompressOp::Decompress)),
    ) else {
        return;
    };

    ratios.push(CustomUnitMeasurement {
        name: format!(
            "vortex:parquet-{} gpu ratio decompress time/{bench_name}",
            options.codec.name()
        ),
        format: Format::OnDiskVortex,
        unit: std::borrow::Cow::from("ratio"),
        value: vortex_time.as_nanos() as f64 / parquet_time.as_nanos() as f64,
    });
}

/// Extracts the message from a caught panic payload.
fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
