// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vector-search-bench` benchmarks for cosine-similarity scan and TurboQuant distortion.
//!
//! ```sh
//! cargo run -p vector-search-bench --release -- search \
//!     --dataset cohere-large-10m \
//!     --layout partitioned \
//!     --flavors vortex-uncompressed,vortex-turboquant \
//!     --iterations 3 \
//!     --threshold 0.8
//!
//! cargo run -p vector-search-bench --release -- distortion \
//!     --dataset sift-small-500k \
//!     --bits 4 \
//!     --samples 4096
//! ```

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use vector_search_bench::compression::ALL_VECTOR_FLAVORS;
use vector_search_bench::compression::VectorFlavor;
use vector_search_bench::display::DatasetReport;
use vector_search_bench::display::render;
use vector_search_bench::distortion::DistortionConfig;
use vector_search_bench::distortion::run_distortion;
use vector_search_bench::prepare::CompressedVortexDataset;
use vector_search_bench::prepare::prepare_all;
use vector_search_bench::query::get_random_query_vector;
use vector_search_bench::resolve_layout;
use vector_search_bench::scan::ScanConfig;
use vector_search_bench::scan::ScanTiming;
use vector_search_bench::scan::run_search_scan;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::v3;
use vortex_bench::vector_dataset;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// On-disk cosine-similarity scan latency benchmark.
    Search(SearchArgs),
    /// TurboQuant distortion measurement: reconstruction error and cosine error.
    Distortion(DistortionArgs),
}

#[derive(Parser, Debug)]
struct SearchArgs {
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

    /// Additionally write v3 JSONL records to this path. See
    /// `benchmarks-website/planning/02-contracts.md`.
    #[arg(long)]
    gh_json_v3: Option<PathBuf>,

    /// Emit verbose tracing.
    #[arg(short, long)]
    verbose: bool,

    /// Enable perfetto tracing output.
    #[arg(long)]
    tracing: bool,
}

#[derive(Parser, Debug)]
struct DistortionArgs {
    /// Dataset to load vectors from. One dataset per invocation.
    #[arg(long, value_enum)]
    dataset: VectorDataset,

    /// Train-split layout. Required when the dataset publishes more than one layout.
    #[arg(long, value_enum)]
    layout: Option<TrainLayout>,

    /// Bits per quantized coordinate.
    #[arg(long, default_value_t = 4)]
    bits: u8,

    /// Seed for the SORF rotation.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of sign-diagonal plus Walsh-Hadamard rounds in the SORF transform.
    #[arg(long, default_value_t = 3)]
    rounds: u8,

    /// Number of base vectors to sample from the first train shard (first N rows).
    #[arg(long, default_value_t = 65536)]
    samples: usize,

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
    let cli = Cli::parse();
    match cli.command {
        Command::Search(args) => run_search(args).await,
        Command::Distortion(args) => run_distortion_cmd(args).await,
    }
}

async fn run_search(args: SearchArgs) -> Result<()> {
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

    let datasets_paths = vector_dataset::download(dataset, layout)
        .await
        .with_context(|| format!("download {}", dataset.name()))?;

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

    if let Some(path) = args.gh_json_v3.as_ref() {
        let records: Vec<v3::V3Record> = scan_timings
            .iter()
            .map(|scan| {
                let all_runs_ns: Vec<u64> = scan
                    .all_runs
                    .iter()
                    .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
                    .collect();
                let median_ns = u64::try_from(scan.median.as_nanos()).unwrap_or(u64::MAX);
                v3::vector_search_record(
                    v3::VectorSearchDims {
                        dataset: dataset.name(),
                        layout: layout.label(),
                        flavor: scan.flavor.label(),
                        threshold: f64::from(args.threshold),
                    },
                    median_ns,
                    all_runs_ns,
                    scan.matches,
                    scan.rows_scanned,
                    scan.bytes_scanned,
                )
            })
            .collect();
        v3::write_jsonl_to_path(path, &records)?;
    }

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

async fn run_distortion_cmd(args: DistortionArgs) -> Result<()> {
    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let layout = resolve_layout(args.dataset, args.layout)?;
    let config = DistortionConfig {
        dataset: args.dataset,
        layout,
        bits: args.bits,
        seed: args.seed,
        rounds: args.rounds,
        samples: args.samples,
    };

    let report = run_distortion(&config).await?;

    if let Some(path) = args.output_path {
        let mut file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        report.render(&mut file)?;
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        report.render(&mut handle)?;
    }

    Ok(())
}
