// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use random_access_bench::AccessPattern;
use random_access_bench::OpenMode;
use random_access_bench::RunConfig;
use vortex_bench::Format;
use vortex_bench::datasets::feature_vectors::FeatureVectorsData;
use vortex_bench::datasets::nested_lists::NestedListsData;
use vortex_bench::datasets::nested_structs::NestedStructsData;
use vortex_bench::datasets::taxi_data::TaxiData;
use vortex_bench::display::DisplayFormat;
use vortex_bench::random_access::BenchDataset;
use vortex_bench::setup_logging_and_tracing;

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
    fn into_dataset(self) -> Box<dyn BenchDataset> {
        match self {
            Self::Taxi => Box::new(TaxiData),
            Self::FeatureVectors => Box::new(FeatureVectorsData),
            Self::NestedLists => Box::new(NestedListsData),
            Self::NestedStructs => Box::new(NestedStructsData),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        value_delimiter = ',',
        value_parser = Format::parse_allowed,
        default_values = ["parquet", "vortex", "lance"]
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
    /// Additionally write benchmark ingest JSONL records to this path.
    #[arg(long = "ingest-jsonl")]
    ingest_output: Option<PathBuf>,
    /// Which datasets to benchmark random access on.
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![DatasetArg::Taxi, DatasetArg::FeatureVectors, DatasetArg::NestedLists, DatasetArg::NestedStructs]
    )]
    datasets: Vec<DatasetArg>,
    /// Which access patterns to benchmark.
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![AccessPattern::Correlated, AccessPattern::Uniform]
    )]
    patterns: Vec<AccessPattern>,
    /// Whether to reopen the file on each iteration, use a cached handle, or run both.
    #[arg(long, value_enum, default_value_t = OpenMode::Both)]
    open_mode: OpenMode,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let run_config = RunConfig {
        datasets: args
            .datasets
            .into_iter()
            .map(DatasetArg::into_dataset)
            .collect(),
        formats: args.formats,
        patterns: args.patterns,
        time_limit: args.time_limit,
        open_mode: args.open_mode,
        display_format: args.display_format,
        output_path: args.output_path,
        ingest_output: args.ingest_output,
    };

    random_access_bench::run(run_config).await
}
