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

use anyhow::Context;
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
use vector_search_bench::verify::VerificationKind;
use vector_search_bench::verify::compute_cosine_scores;
use vector_search_bench::verify::verify_variant;
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
    #[clap(name = "cohere-medium")]
    CohereMedium,
    #[clap(name = "openai-small")]
    OpenAiSmall,
    #[clap(name = "openai-medium")]
    OpenAiMedium,
    #[clap(name = "bioasq-medium")]
    BioasqMedium,
    #[clap(name = "glove-medium")]
    GloveMedium,
}

impl SelectableDataset {
    fn into_dataset(self) -> VectorDataset {
        match self {
            SelectableDataset::CohereSmall => VectorDataset::CohereSmall,
            SelectableDataset::CohereMedium => VectorDataset::CohereMedium,
            SelectableDataset::OpenAiSmall => VectorDataset::OpenAiSmall,
            SelectableDataset::OpenAiMedium => VectorDataset::OpenAiMedium,
            SelectableDataset::BioasqMedium => VectorDataset::BioasqMedium,
            SelectableDataset::GloveMedium => VectorDataset::GloveMedium,
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
    let mut verification: Vec<CustomUnitMeasurement> = Vec::new();

    for dataset in &datasets {
        let prepared = prepare_dataset(dataset).await?;
        tracing::info!(
            "prepared {}: dim={}, num_rows={}",
            prepared.name,
            prepared.dim(),
            prepared.num_rows()
        );

        // Ground-truth cosine scores for the verification query — the scores produced by
        // the uncompressed Vortex scan. Every other variant (including the parquet
        // hand-rolled loop) will be compared against this.
        let baseline_scores =
            compute_cosine_scores(&prepared.uncompressed, &prepared.query, &SESSION)
                .context("compute ground-truth cosine scores for verification")?;
        tracing::info!(
            "computed {} ground-truth cosine scores for {}",
            baseline_scores.len(),
            prepared.name
        );

        // Parquet-Arrow baseline. Emitted as a separate pseudo-variant with label
        // `parquet` / Format::Parquet so it shows up in dashboards next to the Vortex
        // variants. The parquet baseline uses a hand-rolled Rust cosine loop; it must
        // match the Vortex cosine scores within lossless tolerance (f32 ULPs) because
        // it's computing the same math on the same underlying f32 values.
        if run_parquet_baseline {
            let parquet_path = dataset.to_parquet_path().await?;
            let baseline_data =
                vector_search_bench::parquet_baseline::read_parquet_embedding_column(&parquet_path)
                    .context("read parquet emb column for verification")?;
            let parquet_scores = vector_search_bench::parquet_baseline::cosine_loop(
                &baseline_data.elements,
                baseline_data.num_rows,
                baseline_data.dim,
                &prepared.query,
            );
            let parquet_report = vector_search_bench::verify::verify_scores(
                &baseline_scores,
                &parquet_scores,
                VerificationKind::Lossless,
            );
            if !parquet_report.passed {
                anyhow::bail!(
                    "parquet baseline correctness check failed on {}: \
                     max_abs_diff={:.6}, mean_abs_diff={:.6}, tolerance={:.6}",
                    prepared.name,
                    parquet_report.max_abs_diff,
                    parquet_report.mean_abs_diff,
                    parquet_report.tolerance(),
                );
            }
            tracing::info!(
                "parquet/{} verification: max_abs_diff={:.2e}, mean_abs_diff={:.2e}",
                prepared.name,
                parquet_report.max_abs_diff,
                parquet_report.mean_abs_diff,
            );
            verification.push(CustomUnitMeasurement {
                name: format!("correctness-max-diff/parquet/{}", prepared.name),
                format: Format::Parquet,
                unit: Cow::from("abs-diff"),
                value: parquet_report.max_abs_diff,
            });

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
                name: format!("decompress time/{bench_name}"),
                format: Format::Parquet,
                time: baseline_timings.decompress,
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
            let prep = prepare_variant(&prepared, variant, &SESSION)?;

            let variant_label = variant.label();
            let bench_name = format!("{variant_label}/{}", prepared.name);

            // Correctness verification BEFORE timing. Lossless variants must match
            // the uncompressed baseline within f32 noise; TurboQuant must stay within
            // its lossy tolerance. A failure bails the whole run — you cannot publish
            // throughput numbers for an encoding that returns wrong answers.
            let kind = if variant == Variant::VortexTurboQuant {
                VerificationKind::Lossy
            } else {
                VerificationKind::Lossless
            };
            let report = verify_variant(
                &bench_name,
                &prep.array,
                &prepared.query,
                &baseline_scores,
                kind,
                &SESSION,
            )?;
            tracing::info!(
                "{} verification ({:?}): max_abs_diff={:.2e}, mean_abs_diff={:.2e}",
                bench_name,
                kind,
                report.max_abs_diff,
                report.mean_abs_diff,
            );
            verification.push(CustomUnitMeasurement {
                name: format!("correctness-max-diff/{bench_name}"),
                format: variant.as_format(),
                unit: Cow::from("abs-diff"),
                value: report.max_abs_diff,
            });

            // In-memory nbytes — the honest size of the variant tree we're executing.
            sizes.push(CustomUnitMeasurement {
                name: format!("{variant_label} nbytes/{}", prepared.name),
                format: variant.as_format(),
                unit: Cow::from("bytes"),
                value: prep.nbytes as f64,
            });

            // Compress time — the wall time it takes to build the variant tree from
            // the materialized uncompressed source. For the uncompressed variant
            // itself this is ~0 (identity), so we still emit it as a measurement for
            // dashboard consistency.
            timings.push(CompressionTimingMeasurement {
                name: format!("compress time/{bench_name}"),
                format: variant.as_format(),
                time: prep.compress_duration,
            });

            let variant_timings =
                run_timings(&prep.array, &prepared.query, args.iterations, &SESSION)?;

            timings.push(CompressionTimingMeasurement {
                name: format!("decompress time/{bench_name}"),
                format: variant.as_format(),
                time: variant_timings.decompress,
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
            // variants are trivially 1.0 by construction (since they agree with the
            // uncompressed baseline within 1e-4) so we skip them to keep noise down.
            if args.recall_queries > 0 && variant == Variant::VortexTurboQuant {
                let recall = measure_recall_at_k(
                    &prepared.uncompressed,
                    &prep.array,
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
            for check in &verification {
                writeln!(writer, "{} {:.6e} {}", check.name, check.value, check.unit)?;
            }
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, timings)?;
            print_measurements_json(&mut writer, sizes)?;
            print_measurements_json(&mut writer, recalls)?;
            print_measurements_json(&mut writer, verification)?;
        }
    }

    Ok(())
}

use std::io::Write;
