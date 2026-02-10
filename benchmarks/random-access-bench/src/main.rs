// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use clap::Parser;
use clap::ValueEnum;
use indicatif::ProgressBar;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use vortex_bench::BenchmarkOutput;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Target;
use vortex_bench::datasets::feature_vectors::*;
use vortex_bench::datasets::nested_lists::*;
use vortex_bench::datasets::nested_structs::*;
use vortex_bench::datasets::taxi_data::*;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::display::render_table;
use vortex_bench::measurements::TimingMeasurement;
use vortex_bench::random_access::BenchDataset;
use vortex_bench::random_access::ParquetRandomAccessor;
use vortex_bench::random_access::RandomAccessor;
use vortex_bench::random_access::VortexRandomAccessor;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::utils::constants::STORAGE_NVME;

// ---------------------------------------------------------------------------
// Dataset implementations
// ---------------------------------------------------------------------------

/// Short format label used in benchmark measurement names.
fn format_label(format: Format) -> &'static str {
    match format {
        Format::OnDiskVortex => "vortex",
        Format::VortexCompact => "vortex-compact",
        Format::Parquet => "parquet",
        Format::Lance => "lance",
        other => unimplemented!("Random access bench not implemented for {other}"),
    }
}

/// Create a random accessor from a file path, dataset name, and format.
///
/// This eliminates the repeated match-on-format boilerplate in each dataset.
fn create_accessor(path: PathBuf, dataset: &str, format: Format) -> Box<dyn RandomAccessor> {
    let name = format!(
        "random-access/{}/{}-tokio-local-disk",
        dataset,
        format_label(format)
    );
    match format {
        Format::OnDiskVortex | Format::VortexCompact => Box::new(
            VortexRandomAccessor::with_name_and_format(path, name, format),
        ),
        Format::Parquet => Box::new(ParquetRandomAccessor::with_name(path, name)),
        #[cfg(feature = "lance")]
        Format::Lance => {
            use lance_bench::random_access::LanceRandomAccessor;
            Box::new(LanceRandomAccessor::with_name(path, name))
        }
        other => unimplemented!("Random access bench not implemented for {other}"),
    }
}

/// A function returning a boxed future that resolves to a file path.
type PathFn = fn() -> Pin<Box<dyn Future<Output = Result<PathBuf>> + Send>>;

/// Paths for a specific dataset, keyed by format.
struct DatasetPaths {
    name: &'static str,
    row_count: u64,
    parquet: PathFn,
    vortex: PathFn,
    vortex_compact: PathFn,
    #[cfg(feature = "lance")]
    lance: PathFn,
}

#[async_trait]
impl BenchDataset for DatasetPaths {
    fn name(&self) -> &str {
        self.name
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    async fn create(&self, format: Format) -> Result<Box<dyn RandomAccessor>> {
        let path = match format {
            Format::OnDiskVortex => (self.vortex)().await?,
            Format::VortexCompact => (self.vortex_compact)().await?,
            Format::Parquet => (self.parquet)().await?,
            #[cfg(feature = "lance")]
            Format::Lance => (self.lance)().await?,
            other => unimplemented!("Random access bench not implemented for {other}"),
        };
        Ok(create_accessor(path, self.name, format))
    }
}

fn taxi_dataset() -> DatasetPaths {
    DatasetPaths {
        name: "taxi",
        row_count: 3_339_715,
        parquet: || Box::pin(taxi_data_parquet()),
        vortex: || Box::pin(taxi_data_vortex()),
        vortex_compact: || Box::pin(taxi_data_vortex_compact()),
        #[cfg(feature = "lance")]
        lance: || Box::pin(lance_bench::random_access::taxi_data_lance()),
    }
}

fn feature_vectors_dataset() -> DatasetPaths {
    DatasetPaths {
        name: "feature-vectors",
        row_count: 1_000_000,
        parquet: || Box::pin(feature_vectors_parquet()),
        vortex: || Box::pin(feature_vectors_vortex()),
        vortex_compact: || Box::pin(feature_vectors_vortex_compact()),
        #[cfg(feature = "lance")]
        lance: || Box::pin(lance_bench::random_access::feature_vectors_lance()),
    }
}

fn nested_lists_dataset() -> DatasetPaths {
    DatasetPaths {
        name: "nested-lists",
        row_count: 1_000_000,
        parquet: || Box::pin(nested_lists_parquet()),
        vortex: || Box::pin(nested_lists_vortex()),
        vortex_compact: || Box::pin(nested_lists_vortex_compact()),
        #[cfg(feature = "lance")]
        lance: || Box::pin(lance_bench::random_access::nested_lists_lance()),
    }
}

fn nested_structs_dataset() -> DatasetPaths {
    DatasetPaths {
        name: "nested-structs",
        row_count: 1_000_000,
        parquet: || Box::pin(nested_structs_parquet()),
        vortex: || Box::pin(nested_structs_vortex()),
        vortex_compact: || Box::pin(nested_structs_vortex_compact()),
        #[cfg(feature = "lance")]
        lance: || Box::pin(lance_bench::random_access::nested_structs_lance()),
    }
}

// ---------------------------------------------------------------------------
// Access patterns
// ---------------------------------------------------------------------------

/// Access pattern for random access benchmarks.
#[derive(Clone, Copy, Debug)]
enum AccessPattern {
    /// Multiple clusters of sequential indices scattered across the dataset,
    /// simulating workloads with spatial locality (e.g. scanning nearby records).
    Correlated,
    /// Indices generated by a Poisson process (exponential inter-arrival times)
    /// spread uniformly across the dataset, simulating random lookups with no locality.
    Uniform,
}

impl AccessPattern {
    fn name(&self) -> &'static str {
        match self {
            AccessPattern::Correlated => "correlated",
            AccessPattern::Uniform => "uniform",
        }
    }
}

const ACCESS_PATTERNS: [AccessPattern; 2] = [AccessPattern::Correlated, AccessPattern::Uniform];

/// Number of clusters for the correlated pattern.
const NUM_CLUSTERS: usize = 5;

/// Number of consecutive indices per cluster.
const CLUSTER_SIZE: usize = 20;

/// Expected number of indices for the Poisson (uniform) pattern.
const POISSON_EXPECTED_COUNT: usize = 100;

/// Generate indices for the given dataset and access pattern.
fn generate_indices(dataset: &dyn BenchDataset, pattern: AccessPattern) -> Vec<u64> {
    let row_count = dataset.row_count();
    let mut rng = StdRng::seed_from_u64(42);

    match pattern {
        AccessPattern::Correlated => {
            // Pick random cluster starts, then emit CLUSTER_SIZE consecutive indices from each.
            let mut indices = Vec::with_capacity(NUM_CLUSTERS * CLUSTER_SIZE);
            for _ in 0..NUM_CLUSTERS {
                let start = rng.random_range(0..row_count.saturating_sub(CLUSTER_SIZE as u64));
                for offset in 0..CLUSTER_SIZE as u64 {
                    indices.push(start + offset);
                }
            }
            indices.sort_unstable();
            indices
        }
        AccessPattern::Uniform => {
            // Poisson process: exponential inter-arrival times with rate chosen to yield
            // ~POISSON_EXPECTED_COUNT indices across the dataset.
            let rate = POISSON_EXPECTED_COUNT as f64 / row_count as f64;
            // SAFETY: rate is always positive (POISSON_EXPECTED_COUNT > 0, row_count > 0).
            #[allow(clippy::unwrap_used)]
            let exp = Exp::new(rate).unwrap();
            let mut indices = Vec::with_capacity(POISSON_EXPECTED_COUNT);
            let mut pos = 0.0_f64;
            loop {
                let gap: f64 = exp.sample(&mut rng);
                pos += gap;
                #[allow(clippy::cast_possible_truncation)]
                let idx = pos as u64;
                if idx >= row_count {
                    break;
                }
                indices.push(idx);
            }
            indices
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Which synthetic dataset to benchmark.
#[derive(ValueEnum, Clone, Copy, Debug)]
enum DatasetArg {
    #[clap(name = "taxi")]
    Taxi,
    #[clap(name = "feature-vectors")]
    FeatureVectors,
    #[clap(name = "nested-lists")]
    NestedLists,
    #[clap(name = "nested-structs")]
    NestedStructs,
}

impl DatasetArg {
    fn into_dataset(self) -> DatasetPaths {
        match self {
            DatasetArg::Taxi => taxi_dataset(),
            DatasetArg::FeatureVectors => feature_vectors_dataset(),
            DatasetArg::NestedLists => nested_lists_dataset(),
            DatasetArg::NestedStructs => nested_structs_dataset(),
        }
    }
}

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
    /// Time limit in seconds for each benchmark target (e.g., 10 for 10 seconds).
    #[arg(long, default_value_t = 10)]
    time_limit: u64,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short)]
    output_path: Option<PathBuf>,
    /// Which datasets to benchmark random access on.
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![DatasetArg::Taxi, DatasetArg::FeatureVectors, DatasetArg::NestedLists, DatasetArg::NestedStructs]
    )]
    datasets: Vec<DatasetArg>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let datasets: Vec<DatasetPaths> = args
        .datasets
        .into_iter()
        .map(|d| d.into_dataset())
        .collect();

    run_random_access(
        &datasets,
        args.formats,
        args.time_limit,
        args.display_format,
        args.output_path,
    )
    .await
}

// ---------------------------------------------------------------------------
// Benchmark core
// ---------------------------------------------------------------------------

/// Run a random access benchmark for the given accessor (already opened).
///
/// Runs the take operation repeatedly until the time limit is reached,
/// collecting timing for each run.
async fn benchmark_random_access(
    accessor: &dyn RandomAccessor,
    measurement_name: &str,
    indices: &[u64],
    time_limit_secs: u64,
    storage: &str,
) -> Result<TimingMeasurement> {
    let time_limit = Duration::from_secs(time_limit_secs);
    let overall_start = Instant::now();
    let mut runs = Vec::new();

    // Run at least once, then continue until time limit
    loop {
        let start = Instant::now();
        let _row_count = accessor.take(indices).await?;
        runs.push(start.elapsed());

        if overall_start.elapsed() >= time_limit {
            break;
        }
    }

    Ok(TimingMeasurement {
        name: measurement_name.to_string(),
        storage: storage.to_string(),
        target: Target::new(format_to_engine(accessor.format()), accessor.format()),
        runs,
    })
}

/// Build a measurement name for a benchmark run.
///
/// For taxi (legacy), the name is `random-access/{format}-tokio-local-disk` to preserve
/// historical continuity with existing benchmark data.
/// For other datasets, includes dataset and pattern:
/// `random-access/{dataset}/{pattern}/{format}-tokio-local-disk`.
fn measurement_name(dataset: &str, pattern: Option<AccessPattern>, format: Format) -> String {
    match pattern {
        Some(p) => format!(
            "random-access/{}/{}/{}-tokio-local-disk",
            dataset,
            p.name(),
            format_label(format)
        ),
        None => format!("random-access/{}-tokio-local-disk", format_label(format)),
    }
}

/// Map format to the appropriate engine for random access benchmarks.
fn format_to_engine(format: Format) -> Engine {
    match format {
        Format::OnDiskVortex | Format::VortexCompact => Engine::Vortex,
        Format::Parquet => Engine::Arrow,
        #[cfg(feature = "lance")]
        Format::Lance => Engine::Arrow, // Is this right here?
        _ => Engine::default(),
    }
}

/// The benchmark ID used for output path.
const BENCHMARK_ID: &str = "random-access";

/// Fixed indices used by the original taxi benchmark (preserved for historical continuity).
const FIXED_TAXI_INDICES: [u64; 6] = [10, 11, 12, 13, 100_000, 3_000_000];

async fn run_random_access(
    datasets: &[DatasetPaths],
    formats: Vec<Format>,
    time_limit: u64,
    display_format: DisplayFormat,
    output_path: Option<PathBuf>,
) -> Result<()> {
    let total_steps: usize = datasets
        .iter()
        .map(|d| {
            let legacy_extra = if d.name() == "taxi" { formats.len() } else { 0 };
            formats.len() * ACCESS_PATTERNS.len() + legacy_extra
        })
        .sum();
    let progress = ProgressBar::new(total_steps as u64);

    let mut targets = Vec::new();
    let mut measurements = Vec::new();

    for dataset in datasets {
        for format in &formats {
            if dataset.name() == "taxi" {
                let mut accessor = dataset.create(*format).await?;
                accessor.open().await?;
                let name = measurement_name(dataset.name(), None, *format);
                let measurement = benchmark_random_access(
                    accessor.as_ref(),
                    &name,
                    &FIXED_TAXI_INDICES,
                    time_limit,
                    STORAGE_NVME,
                )
                .await?;

                targets.push(measurement.target);
                measurements.push(measurement);
                progress.inc(1);
            }

            for pattern in &ACCESS_PATTERNS {
                let mut accessor = dataset.create(*format).await?;
                accessor.open().await?;
                let indices = generate_indices(dataset, *pattern);
                let name = measurement_name(dataset.name(), Some(*pattern), *format);
                let measurement = benchmark_random_access(
                    accessor.as_ref(),
                    &name,
                    &indices,
                    time_limit,
                    STORAGE_NVME,
                )
                .await?;

                targets.push(measurement.target);
                measurements.push(measurement);
                progress.inc(1);
            }
        }
    }

    progress.finish();

    let output = BenchmarkOutput::with_path(BENCHMARK_ID, output_path);
    let mut writer = output.create_writer()?;

    match display_format {
        DisplayFormat::Table => {
            render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements)?;
        }
    }

    Ok(())
}
