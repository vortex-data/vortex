// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vector-search-bench` — brute-force cosine-similarity benchmark over public VectorDBBench
//! embedding corpora.
//!
//! Usage:
//!
//! ```bash
//! cargo run -p vector-search-bench --release -- \
//!     --datasets cohere-small \
//!     --variants vortex-uncompressed,vortex-default,vortex-turboquant \
//!     --iterations 5 \
//!     -d table
//! ```

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use indicatif::ProgressBar;
use vector_search_bench::DEFAULT_THRESHOLD;
use vector_search_bench::Variant;
use vector_search_bench::parquet_baseline::run_parquet_baseline_timings;
use vector_search_bench::prepare_dataset;
use vector_search_bench::prepare_variant;
use vector_search_bench::recall::DEFAULT_TOP_K;
use vector_search_bench::recall::measure_recall_at_k;
use vector_search_bench::run_timings;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::create_output_writer;
use vortex_bench::datasets::Dataset;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::measurements::CompressionTimingMeasurement;
use vortex_bench::measurements::CustomUnitMeasurement;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::vector_dataset::VectorDataset;

const BENCHMARK_ID: &str = "vector-search";

/// Command-line arguments for `vector-search-bench`.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of timed iterations per measurement. The reported time is the minimum across
    /// iterations (matches compress-bench convention).
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    /// Subset of datasets to run. Defaults to Cohere-small.
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![SelectableDataset::CohereSmall])]
    datasets: Vec<SelectableDataset>,

    /// Which benchmark variants to run, using kebab-cased labels. The `--formats` name is
    /// used (instead of `--variants`) so this benchmark matches the CI invocation
    /// convention shared across random-access-bench / compress-bench. Accepted values:
    /// `parquet`, `vortex-uncompressed`, `vortex-default`, `vortex-turboquant`. Defaults
    /// to running all four.
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![SelectableFormat::Parquet, SelectableFormat::VortexUncompressed, SelectableFormat::VortexDefault, SelectableFormat::VortexTurboQuant])]
    formats: Vec<SelectableFormat>,

    /// Number of query rows sampled when computing Recall@K for TurboQuant. 0 disables
    /// the quality measurement entirely (useful for smoke tests).
    #[arg(long, default_value_t = 100)]
    recall_queries: usize,

    /// K in Recall@K. Defaults to 10, matching VectorDBBench conventions.
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    recall_k: usize,

    /// Output display format (`table` for humans, `gh-json` for CI ingestion).
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    /// If set, write output to this file instead of stdout.
    #[arg(short, long)]
    output_path: Option<PathBuf>,

    /// Verbose logging.
    #[arg(short, long)]
    verbose: bool,

    /// Enable perfetto tracing output.
    #[arg(long)]
    tracing: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum SelectableDataset {
    #[clap(name = "cohere-small")]
    CohereSmall,
}

impl SelectableDataset {
    fn into_dataset(self) -> VectorDataset {
        match self {
            SelectableDataset::CohereSmall => VectorDataset::CohereSmall,
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum SelectableFormat {
    /// Parquet-Arrow hand-rolled cosine loop baseline.
    #[clap(name = "parquet")]
    Parquet,
    /// Raw `Vector<dim, f32>` with no encoding compression.
    #[clap(name = "vortex-uncompressed")]
    VortexUncompressed,
    /// BtrBlocks default-compression applied to the FSL storage child.
    #[clap(name = "vortex-default")]
    VortexDefault,
    /// Full TurboQuant pipeline (lossy).
    #[clap(name = "vortex-turboquant")]
    VortexTurboQuant,
}

impl SelectableFormat {
    fn into_variant(self) -> Option<Variant> {
        match self {
            SelectableFormat::Parquet => None,
            SelectableFormat::VortexUncompressed => Some(Variant::VortexUncompressed),
            SelectableFormat::VortexDefault => Some(Variant::VortexDefault),
            SelectableFormat::VortexTurboQuant => Some(Variant::VortexTurboQuant),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let datasets: Vec<VectorDataset> = args
        .datasets
        .iter()
        .copied()
        .map(SelectableDataset::into_dataset)
        .collect();

    let run_parquet_baseline = args.formats.contains(&SelectableFormat::Parquet);
    let variants: Vec<Variant> = args
        .formats
        .iter()
        .filter_map(|f| f.into_variant())
        .collect();

    let total_work = datasets.len() * args.formats.len();
    let progress = ProgressBar::new(total_work as u64);

    let mut timings: Vec<CompressionTimingMeasurement> = Vec::new();
    let mut sizes: Vec<CustomUnitMeasurement> = Vec::new();

    let mut recalls: Vec<CustomUnitMeasurement> = Vec::new();

    for dataset in &datasets {
        let prepared = prepare_dataset(dataset).await?;
        tracing::info!(
            "prepared {}: dim={}, num_rows={}",
            prepared.name,
            prepared.dim(),
            prepared.num_rows()
        );

        // Parquet-Arrow baseline. Emitted as a separate pseudo-variant with label
        // `parquet` / Format::Parquet so it shows up in dashboards next to the Vortex
        // variants.
        if run_parquet_baseline {
            let parquet_path = dataset.to_parquet_path().await?;
            let baseline_timings = run_parquet_baseline_timings(
                &parquet_path,
                &prepared.query,
                DEFAULT_THRESHOLD,
                args.iterations,
            )?;

            let label = "parquet";
            let bench_name = format!("{label}/{}", prepared.name);

            sizes.push(CustomUnitMeasurement {
                name: format!("{label} size/{}", prepared.name),
                format: Format::Parquet,
                unit: Cow::from("bytes"),
                value: prepared.parquet_bytes as f64,
            });
            timings.push(CompressionTimingMeasurement {
                name: format!("decode time/{bench_name}"),
                format: Format::Parquet,
                time: baseline_timings.decode,
            });
            timings.push(CompressionTimingMeasurement {
                name: format!("cosine-similarity time/{bench_name}"),
                format: Format::Parquet,
                time: baseline_timings.cosine,
            });
            timings.push(CompressionTimingMeasurement {
                name: format!("cosine-filter time/{bench_name}"),
                format: Format::Parquet,
                time: baseline_timings.filter,
            });
        }

        for &variant in &variants {
            let (variant_array, size_bytes) = prepare_variant(&prepared, variant, &SESSION).await?;

            let variant_label = variant.label();
            let bench_name = format!("{variant_label}/{}", prepared.name);

            sizes.push(CustomUnitMeasurement {
                name: format!("{variant_label} size/{}", prepared.name),
                format: variant.as_format(),
                unit: Cow::from("bytes"),
                value: size_bytes as f64,
            });

            let variant_timings =
                run_timings(&variant_array, &prepared.query, args.iterations, &SESSION)?;

            timings.push(CompressionTimingMeasurement {
                name: format!("decode time/{bench_name}"),
                format: variant.as_format(),
                time: variant_timings.decode,
            });
            timings.push(CompressionTimingMeasurement {
                name: format!("cosine-similarity time/{bench_name}"),
                format: variant.as_format(),
                time: variant_timings.cosine,
            });
            timings.push(CompressionTimingMeasurement {
                name: format!("cosine-filter time/{bench_name}"),
                format: variant.as_format(),
                time: variant_timings.filter,
            });

            // Recall@K quality measurement for lossy variants only. The lossless
            // variants (uncompressed + BtrBlocks default) are trivially 1.0 against
            // the uncompressed ground truth, so we skip them to avoid noise.
            if args.recall_queries > 0 && variant == Variant::VortexTurboQuant {
                let recall = measure_recall_at_k(
                    &prepared.uncompressed,
                    &variant_array,
                    args.recall_queries,
                    args.recall_k,
                    &SESSION,
                )?;
                tracing::info!("Recall@{} for {}: {:.4}", args.recall_k, bench_name, recall);
                recalls.push(CustomUnitMeasurement {
                    name: format!("recall@{}/{bench_name}", args.recall_k),
                    format: variant.as_format(),
                    unit: Cow::from("recall"),
                    value: recall,
                });
            }

            progress.inc(1);
        }
    }
    progress.finish();

    let mut writer = create_output_writer(&args.display_format, args.output_path, BENCHMARK_ID)?;
    match args.display_format {
        DisplayFormat::Table => {
            // Our variants span multiple `Format` values *and* multiple labels that share a
            // single `Format`, so the existing `render_table` helper (which groups by
            // `Target`) would collapse them. Emit one line per measurement instead; this is
            // only used for developer inspection — CI consumes `gh-json` via the arm below.
            for timing in &timings {
                writeln!(writer, "{} {} ns", timing.name, timing.time.as_nanos())?;
            }
            for size in &sizes {
                writeln!(writer, "{} {} {}", size.name, size.value, size.unit)?;
            }
            for recall in &recalls {
                writeln!(
                    writer,
                    "{} {:.4} {}",
                    recall.name, recall.value, recall.unit
                )?;
            }
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, timings)?;
            print_measurements_json(&mut writer, sizes)?;
            print_measurements_json(&mut writer, recalls)?;
        }
    }

    Ok(())
}

use std::io::Write;
