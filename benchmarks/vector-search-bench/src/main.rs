// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vector-search-bench` — on-disk cosine-similarity scan benchmark.
//!
//! ```sh
//! cargo run -p vector-search-bench --release -- \
//!     --dataset cohere-large-10m \
//!     --layout partitioned \
//!     --flavors vortex-uncompressed,vortex-turboquant,handrolled \
//!     --iterations 3 \
//!     --threshold 0.8
//! ```

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use vector_search_bench::compression::VortexCompression;
use vector_search_bench::display::DatasetReport;
use vector_search_bench::display::render;
use vector_search_bench::handrolled::run_handrolled_scan;
use vector_search_bench::prepare::CompressionResult;
use vector_search_bench::prepare::prepare_all;
use vector_search_bench::query::sample_query;
use vector_search_bench::recall::RecallConfig;
use vector_search_bench::recall::RecallResult;
use vector_search_bench::recall::measure_recall;
use vector_search_bench::scan::ScanConfig;
use vector_search_bench::scan::ScanTiming;
use vector_search_bench::scan::run_scan;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_bench::vector_dataset::download::download;
use vortex_bench::vector_dataset::layout::TrainLayout;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset to benchmark. Single dataset per CLI invocation by design — large datasets
    /// are intentionally babysat one at a time.
    #[arg(long, value_enum)]
    dataset: VectorDataset,

    /// Train-split layout. Required when the dataset publishes more than one layout.
    /// Defaults to the catalog's first hosted layout when omitted.
    #[arg(long, value_enum)]
    layout: Option<TrainLayout>,

    /// Comma-separated list of flavors to run. Each Vortex flavor produces one `.vortex`
    /// file per train shard; `handrolled` reads the parquet shards directly.
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![FlavorArg::VortexUncompressed, FlavorArg::VortexTurboquant, FlavorArg::Handrolled],
    )]
    flavors: Vec<FlavorArg>,

    /// Number of timed scan iterations per flavor. Best-of-N is reported as the headline.
    #[arg(long, default_value_t = 3, value_parser = parse_positive_usize)]
    iterations: usize,

    /// Cosine threshold passed to the filter expression.
    #[arg(long, default_value_t = 0.8)]
    threshold: f32,

    /// Seed for the test-parquet query sampler.
    #[arg(long, default_value_t = 42)]
    query_seed: u64,

    /// Measure Recall@K for lossy flavors against `neighbors.parquet`. Bails if the
    /// dataset doesn't host neighbors.
    #[arg(long, default_value_t = false)]
    recall: bool,

    /// Number of query rows sampled when computing Recall@K. Distinct from --query-seed
    /// so the recall sampler can pick a different seeded set.
    #[arg(long, default_value_t = 100, value_parser = parse_positive_usize)]
    recall_queries: usize,

    /// K in Recall@K. Defaults to 10 (matches VectorDBBench convention).
    #[arg(long, default_value_t = 10, value_parser = parse_positive_usize)]
    recall_k: usize,

    /// Seed for the recall query sampler. Distinct from --query-seed so the throughput
    /// scan and the recall pass can pick non-correlated query sets.
    #[arg(long, default_value_t = 1234)]
    recall_seed: u64,

    /// Optional path to write the rendered table to instead of stdout.
    #[arg(long)]
    output_path: Option<PathBuf>,

    /// Emit verbose tracing.
    #[arg(short, long)]
    verbose: bool,

    /// Enable perfetto tracing output.
    #[arg(long)]
    tracing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FlavorArg {
    #[clap(name = "vortex-uncompressed")]
    VortexUncompressed,
    #[clap(name = "vortex-turboquant")]
    VortexTurboquant,
    #[clap(name = "handrolled")]
    Handrolled,
}

impl FlavorArg {
    fn to_vortex(self) -> Option<VortexCompression> {
        match self {
            FlavorArg::VortexUncompressed => Some(VortexCompression::Uncompressed),
            FlavorArg::VortexTurboquant => Some(VortexCompression::TurboQuant),
            FlavorArg::Handrolled => None,
        }
    }
}

fn parse_positive_usize(s: &str) -> std::result::Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|e| format!("invalid integer '{s}': {e}"))?;
    if n == 0 {
        return Err("value must be >= 1".to_string());
    }
    Ok(n)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let dataset = args.dataset;
    let layout = resolve_layout(dataset, args.layout)?;
    tracing::info!(
        "running {} on layout {} ({} dims, {} rows)",
        dataset.name(),
        layout,
        dataset.dim(),
        dataset.num_rows()
    );

    let paths = download(dataset, layout)
        .await
        .with_context(|| format!("download {}", dataset.name()))?;

    let vortex_flavors: Vec<VortexCompression> =
        args.flavors.iter().filter_map(|f| f.to_vortex()).collect();
    let run_handrolled = args.flavors.contains(&FlavorArg::Handrolled);
    if vortex_flavors.is_empty() && !run_handrolled {
        anyhow::bail!("no flavors selected — pass at least one to --flavors");
    }

    let prepared = if vortex_flavors.is_empty() {
        Vec::new()
    } else {
        prepare_all(dataset, layout, &vortex_flavors, &paths).await?
    };

    let query_sample = sample_query(
        &paths.test,
        dataset.dim(),
        dataset.element_ptype(),
        args.query_seed,
    )
    .await?;
    tracing::info!(
        "sampled query row {} (dim={})",
        query_sample.query_row_idx,
        query_sample.dim
    );

    let scan_config = ScanConfig {
        iterations: args.iterations,
        threshold: args.threshold,
    };

    let mut scan_timings: Vec<ScanTiming> = Vec::with_capacity(prepared.len());
    for prep in &prepared {
        let timing = run_scan(prep, &query_sample.query, &scan_config).await?;
        scan_timings.push(timing);
    }

    let handrolled_timing = run_handrolled
        .then(|| {
            run_handrolled_scan(
                &paths.train_files,
                &query_sample.query,
                args.threshold,
                args.iterations,
            )
        })
        .transpose()?;

    let recall_results = if args.recall {
        let neighbors_path = paths.neighbors.as_ref().with_context(|| {
            format!(
                "--recall requested but dataset {} does not host neighbors.parquet",
                dataset.name()
            )
        })?;
        let recall_config = RecallConfig {
            k: args.recall_k,
            num_queries: args.recall_queries,
            query_seed: args.recall_seed,
        };
        let mut out: Vec<RecallResult> = Vec::with_capacity(prepared.len());
        for prep in &prepared {
            // Lossless flavors are trivially 1.0; only TurboQuant needs measurement.
            if prep.flavor == VortexCompression::Uncompressed {
                tracing::info!(
                    "skipping recall for lossless flavor {} (trivially 1.0)",
                    prep.flavor.label()
                );
                continue;
            }
            let r = measure_recall(
                prep,
                &paths.test,
                neighbors_path,
                dataset.element_ptype(),
                &recall_config,
            )
            .await?;
            tracing::info!(
                "recall@{} for {}: mean={:.4}, p05={:.4}",
                r.k,
                r.flavor.label(),
                r.mean_recall,
                r.p05_recall,
            );
            out.push(r);
        }
        out
    } else {
        Vec::new()
    };

    let pairs: Vec<(VortexCompression, &CompressionResult, &ScanTiming)> = prepared
        .iter()
        .zip(scan_timings.iter())
        .map(|(prep, scan)| (prep.flavor, prep, scan))
        .collect();
    let report = DatasetReport {
        dataset_name: dataset.name(),
        vortex_results: &pairs,
        handrolled: handrolled_timing.as_ref(),
        recall: &recall_results,
    };

    if let Some(path) = args.output_path {
        let mut file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        render(&report, &mut file)?;
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        render(&report, &mut handle)?;
    }

    Ok(())
}

fn resolve_layout(dataset: VectorDataset, requested: Option<TrainLayout>) -> Result<TrainLayout> {
    let layouts = dataset.layouts();
    match requested {
        Some(layout) => {
            dataset.validate_layout(layout)?;
            Ok(layout)
        }
        None => {
            if layouts.len() == 1 {
                Ok(layouts[0].layout)
            } else {
                let allowed = layouts
                    .iter()
                    .map(|s| s.layout.label())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!(
                    "dataset {} hosts multiple layouts ([{}]) — pass --layout to pick one",
                    dataset.name(),
                    allowed,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn rejects_zero_iterations() {
        let err = Args::try_parse_from([
            "vector-search-bench",
            "--dataset",
            "cohere-small-100k",
            "--iterations",
            "0",
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("value must be >= 1"), "{err}");
    }

    #[test]
    fn parses_layout_argument() {
        let args = Args::try_parse_from([
            "vector-search-bench",
            "--dataset",
            "cohere-large-10m",
            "--layout",
            "partitioned-shuffled",
        ])
        .unwrap();
        assert_eq!(args.layout, Some(TrainLayout::PartitionedShuffled));
    }

    #[test]
    fn defaults_to_all_flavors() {
        let args = Args::try_parse_from(["vector-search-bench", "--dataset", "cohere-small-100k"])
            .unwrap();
        // Two Vortex flavors + handrolled.
        assert_eq!(args.flavors.len(), 3);
    }
}
