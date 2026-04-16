// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vector-search-bench` — on-disk cosine-similarity scan benchmark.
//!
//! ```sh
//! cargo run -p vector-search-bench --release -- \
//!     --dataset cohere-large-10m \
//!     --layout partitioned \
//!     --flavors vortex-uncompressed,vortex-turboquant \
//!     --iterations 3 \
//!     --threshold 0.8
//! ```

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use vector_search_bench::compression::ALL_VECTOR_FLAVORS;
use vector_search_bench::compression::VectorFlavor;
use vector_search_bench::display::DatasetReport;
use vector_search_bench::display::render;
use vector_search_bench::prepare::CompressedVortexDataset;
use vector_search_bench::prepare::prepare_all;
use vector_search_bench::query::get_random_query_vector;
use vector_search_bench::scan::ScanConfig;
use vector_search_bench::scan::ScanTiming;
use vector_search_bench::scan::run_search_scan;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::vector_dataset;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;

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

    /// Comma-separated list of flavors to run. Each Vortex flavor produces one `.vortex` file per
    /// train shard.
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = ALL_VECTOR_FLAVORS.to_vec(),
    )]
    flavors: Vec<VectorFlavor>,

    /// Number of timed scan iterations per flavor. Mean and median are reported.
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Cosine threshold passed to the filter expression.
    #[arg(long, default_value_t = 0.85)]
    threshold: f32,

    /// Seed for the test-parquet query sampler.
    #[arg(long, default_value_t = 42)]
    query_seed: u64,

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

    if args.flavors.is_empty() {
        anyhow::bail!("no flavors selected, please pass at least one to --flavors");
    }

    // Load the source embeddings parquet files.
    let datasets_paths = vector_dataset::download(dataset, layout)
        .await
        .with_context(|| format!("download {}", dataset.name()))?;

    // Load all vortex files needed, compressing new ones if needed.
    let prepared = prepare_all(dataset, layout, &datasets_paths, &args.flavors).await?;

    let query_vector = get_random_query_vector(
        &datasets_paths.test,
        dataset.dim(),
        dataset.element_ptype(),
        args.query_seed,
    )
    .await?;
    tracing::info!(
        "sampled query id {} (dim={})",
        query_vector.id,
        query_vector.query.len()
    );

    let scan_config = ScanConfig {
        iterations: args.iterations,
        threshold: args.threshold,
    };

    // Run all scans and record how long each takes.
    let mut scan_timings: Vec<ScanTiming> = Vec::with_capacity(prepared.len());
    for prep in &prepared {
        let timing = run_search_scan(prep, &query_vector.query, &scan_config).await?;
        scan_timings.push(timing);
    }

    let pairs: Vec<(VectorFlavor, &CompressedVortexDataset, &ScanTiming)> = prepared
        .iter()
        .zip(scan_timings.iter())
        .map(|(prep, scan)| (prep.flavor, prep, scan))
        .collect();
    let report = DatasetReport {
        dataset_name: dataset.name(),
        vortex_results: &pairs,
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
                Ok(layouts[0].layout())
            } else {
                let allowed = layouts
                    .iter()
                    .map(|s| s.layout().label())
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
