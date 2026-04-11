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
//!     --formats handrolled,vortex-uncompressed,vortex-default,vortex-turboquant \
//!     --iterations 5 \
//!     -d table
//! ```
//!
//! The `handrolled` variant is a hand-rolled Rust scalar cosine loop over a flat
//! `Vec<f32>` decoded from the dataset's canonical parquet file; it is a compute-cost
//! floor, not a realistic parquet-on-DBMS baseline. See
//! [`handrolled_baseline`](vector_search_bench::handrolled_baseline) for details.

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use indicatif::ProgressBar;
use vector_search_bench::Variant;
use vector_search_bench::display::build_table_rows;
use vector_search_bench::display::render_variants_table;
use vector_search_bench::handrolled_baseline::run_handrolled_and_collect;
use vector_search_bench::prepare_dataset;
use vector_search_bench::prepare_variant;
use vector_search_bench::recall::DEFAULT_TOP_K;
use vector_search_bench::recall::measure_recall_at_k;
use vector_search_bench::run_timings;
use vector_search_bench::verify::VerificationKind;
use vector_search_bench::verify::compute_cosine_scores;
use vector_search_bench::verify::verify_variant;
use vortex_bench::create_output_writer;
use vortex_bench::datasets::Dataset;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::measurements::CompressionTimingMeasurement;
use vortex_bench::measurements::CustomUnitMeasurement;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::vector_dataset::VectorDataset;

const BENCHMARK_ID: &str = "vector-search";

fn parse_positive_usize(s: &str) -> std::result::Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|err| format!("invalid integer '{s}': {err}"))?;
    if value == 0 {
        return Err("value must be >= 1".to_string());
    }
    Ok(value)
}

/// Command-line arguments for `vector-search-bench`.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of timed iterations per measurement. The reported time is the minimum across
    /// iterations (matches compress-bench convention). Must be >= 1.
    #[arg(short, long, default_value_t = 5, value_parser = parse_positive_usize)]
    iterations: usize,

    /// Subset of datasets to run. Defaults to Cohere-small.
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![SelectableDataset::CohereSmall])]
    datasets: Vec<SelectableDataset>,

    /// Which benchmark variants to run, using kebab-cased labels. The `--formats` name is
    /// used (instead of `--variants`) so this benchmark matches the CI invocation
    /// convention shared across random-access-bench / compress-bench. Accepted values:
    /// `handrolled`, `vortex-uncompressed`, `vortex-default`, `vortex-turboquant`.
    /// Defaults to running all four.
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![SelectableFormat::Handrolled, SelectableFormat::VortexUncompressed, SelectableFormat::VortexDefault, SelectableFormat::VortexTurboQuant])]
    formats: Vec<SelectableFormat>,

    /// Number of query rows sampled when computing Recall@K for TurboQuant. 0 disables
    /// the quality measurement entirely (useful for smoke tests).
    #[arg(long, default_value_t = 100)]
    recall_queries: usize,

    /// K in Recall@K. Defaults to 10, matching VectorDBBench conventions. Must be >= 1.
    #[arg(long, default_value_t = DEFAULT_TOP_K, value_parser = parse_positive_usize)]
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
    /// Hand-rolled Rust scalar cosine loop over a flat `Vec<f32>` decoded from the
    /// canonical parquet file via `parquet-rs` / `arrow-rs`. Compute-cost floor —
    /// not a realistic parquet-on-DBMS baseline. See
    /// [`vector_search_bench::handrolled_baseline`].
    #[clap(name = "handrolled")]
    Handrolled,
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
            SelectableFormat::Handrolled => None,
            SelectableFormat::VortexUncompressed => Some(Variant::VortexUncompressed),
            SelectableFormat::VortexDefault => Some(Variant::VortexDefault),
            SelectableFormat::VortexTurboQuant => Some(Variant::VortexTurboQuant),
        }
    }

    /// Stable kebab-cased label for this variant, used as both the
    /// `--formats` CLI value and the column label in the `-d table`
    /// output. Must match the `#[clap(name = ...)]` attribute on each
    /// enum variant — they sit adjacent so they can't drift.
    fn label(self) -> &'static str {
        match self {
            SelectableFormat::Handrolled => "handrolled",
            SelectableFormat::VortexUncompressed => "vortex-uncompressed",
            SelectableFormat::VortexDefault => "vortex-default",
            SelectableFormat::VortexTurboQuant => "vortex-turboquant",
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

    let run_handrolled_baseline = args.formats.contains(&SelectableFormat::Handrolled);
    let variants: Vec<Variant> = args
        .formats
        .iter()
        .filter_map(|f| f.into_variant())
        .collect();

    // One progress unit per inner-loop body: each Vortex variant plus the handrolled
    // path (when it's enabled) gets exactly one `progress.inc(1)` below. Keep this
    // count in sync with the number of `progress.inc` sites.
    let total_work = datasets.len() * (variants.len() + usize::from(run_handrolled_baseline));
    let progress = ProgressBar::new(total_work as u64);

    let mut timings: Vec<CompressionTimingMeasurement> = Vec::new();
    let mut sizes: Vec<CustomUnitMeasurement> = Vec::new();
    let mut recalls: Vec<CustomUnitMeasurement> = Vec::new();
    let mut verification: Vec<CustomUnitMeasurement> = Vec::new();
    // Dataset names used in metric strings — populated as the outer loop runs
    // so the `-d table` row-construction pass knows exactly which datasets to
    // emit rows for (and in which order). Kept separate from the measurement
    // vecs because the table-row builder looks up measurements by the exact
    // `prepared.name` string the push-side code used.
    let mut dataset_names: Vec<String> = Vec::with_capacity(datasets.len());

    for dataset in &datasets {
        let prepared = prepare_dataset(dataset).await?;
        dataset_names.push(prepared.name.clone());
        tracing::info!(
            "prepared {}: dim={}, num_rows={}",
            prepared.name,
            prepared.dim(),
            prepared.num_rows()
        );
        if args.recall_queries > 0 && args.recall_k > prepared.num_rows() {
            anyhow::bail!(
                "--recall-k {} exceeds dataset '{}' row count {}",
                args.recall_k,
                prepared.name,
                prepared.num_rows()
            );
        }

        // Ground-truth cosine scores for the verification query — the scores produced by
        // the uncompressed Vortex scan. Every other variant (including the hand-rolled
        // baseline) will be compared against this.
        // Ground-truth scores are computed on the f32-cast data so they are directly
        // comparable to the compressed-variant scores (which are always f32). The cast
        // error is part of the measurement -- it reflects the total precision loss a user
        // would see when using f32 encodings on an f64 dataset.
        let baseline_scores =
            compute_cosine_scores(&prepared.uncompressed_f32, &prepared.query_f32)
                .context("compute ground-truth cosine scores for verification")?;
        tracing::info!(
            "computed {} ground-truth cosine scores for {}",
            baseline_scores.len(),
            prepared.name
        );

        // Hand-rolled baseline. Emitted as a separate pseudo-variant with label
        // `handrolled` so it shows up in dashboards next to the Vortex variants.
        // `target.format` stays `Format::Parquet` because the *storage* side is
        // still parquet on disk — only the *compute* is hand-rolled. The metric
        // `name` field carries the `handrolled` label so human readers can tell
        // the compute apart from, say, a DuckDB `list_cosine_similarity`
        // baseline on the same parquet. See
        // [`handrolled_baseline::run_handrolled_and_collect`] for the full
        // timing / verification / push pipeline — kept in the module that
        // defines the baseline so this loop stays focused on dataset iteration.
        if run_handrolled_baseline {
            let parquet_path = dataset.to_parquet_path().await?;
            run_handrolled_and_collect(
                &parquet_path,
                &prepared.name,
                prepared.parquet_bytes,
                &prepared.query_f32,
                &baseline_scores,
                args.iterations,
                &mut timings,
                &mut sizes,
                &mut verification,
            )?;
            progress.inc(1);
        }

        for &variant in &variants {
            let prep = prepare_variant(&prepared, variant)?;

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
                &prepared.query_f32,
                &baseline_scores,
                kind,
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

            let variant_timings = run_timings(
                &prep.array,
                &prepared.query_f32,
                vector_search_bench::DEFAULT_THRESHOLD,
                args.iterations,
            )?;

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
                    &prepared.uncompressed_f32,
                    &prep.array,
                    args.recall_queries,
                    args.recall_k,
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
            // `vector_search_bench::display::render_variants_table` groups
            // columns by **variant label** rather than by `Target`, because
            // multiple vector-search variants legitimately share a single
            // `Format` (e.g. `vortex-uncompressed` and `vortex-default` both
            // map to `Format::OnDiskVortex`). The generic
            // `vortex_bench::display::render_table` helper groups by
            // `Target`, which would collapse those variants into one
            // column, so we render locally instead. The `DisplayFormat::GhJson`
            // arm below is untouched — CI still consumes gh-json byte-for-byte
            // identically.
            //
            // `vortex-uncompressed` is used as the ratio baseline (when
            // present in the run) so that `handrolled` legitimately renders
            // as faster-than-baseline and vortex-default / vortex-turboquant
            // render as ratios of the raw Vortex cost. When the user runs
            // without `vortex-uncompressed`, the renderer falls back to the
            // first column with a value.
            let variant_labels: Vec<String> =
                args.formats.iter().map(|f| f.label().to_owned()).collect();
            let rows = build_table_rows(
                &variant_labels,
                &dataset_names,
                args.recall_k,
                &timings,
                &sizes,
                &recalls,
                &verification,
            );
            render_variants_table(
                &mut writer,
                &variant_labels,
                &rows,
                Some(SelectableFormat::VortexUncompressed.label()),
            )?;
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Args;

    #[test]
    fn args_reject_zero_iterations() {
        let err = Args::try_parse_from(["vector-search-bench", "--iterations", "0"])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("value must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn args_reject_zero_recall_k() {
        let err = Args::try_parse_from(["vector-search-bench", "--recall-k", "0"])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("value must be >= 1"),
            "unexpected error: {err}"
        );
    }
}
