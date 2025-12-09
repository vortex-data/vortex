// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Write;
use std::io::stdout;
use std::path::PathBuf;
use std::time::Instant;

use bench_vortex::BenchmarkDataset;
use bench_vortex::Format;
use bench_vortex::Target;
use bench_vortex::df;
use bench_vortex::display::DisplayFormat;
use bench_vortex::display::print_measurements_json;
use bench_vortex::display::render_table;
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::MetricsSetExt;
use bench_vortex::public_bi::FileType;
use bench_vortex::public_bi::PBI_DATASETS;
use bench_vortex::public_bi::PBIDataset;
use bench_vortex::setup_logging_and_tracing;
use bench_vortex::utils::constants::STORAGE_NVME;
use bench_vortex::utils::new_tokio_runtime;
use clap::Parser;
use clap::value_parser;
use indicatif::ProgressBar;
use itertools::Itertools;
use tracing::Instrument;
use tracing::info_span;
use vortex::error::VortexExpect;
use vortex::error::vortex_panic;
use vortex_datafusion::metrics::VortexMetricsFinder;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long)]
    display_metrics: bool,
    #[arg(long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,
    #[arg(short, long, value_delimiter = ',')]
    dataset: PBIDataset,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(short)]
    output_path: Option<PathBuf>,
    #[arg(long)]
    tracing: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let runtime = new_tokio_runtime(args.threads)?;

    let pbi_dataset = PBI_DATASETS.get(args.dataset);
    let queries = match args.queries.clone() {
        None => pbi_dataset.queries()?,
        Some(queries) => pbi_dataset
            .queries()?
            .into_iter()
            .filter(|(q_idx, _)| queries.iter().contains(q_idx))
            .collect(),
    };

    let progress_bar = ProgressBar::new((queries.len() * args.targets.len()) as u64);
    let mut all_measurements = Vec::default();
    let mut metrics = Vec::new();

    let dataset = pbi_dataset.dataset()?;
    tracing::info!("preparing files");
    // download csvs, unzip, convert to parquet, and convert that to vortex
    runtime.block_on(dataset.write_as_vortex())?;

    for target in &args.targets {
        let format = target.format();
        let session = df::get_session_context(args.disable_datafusion_cache);

        let file_type = match format {
            Format::Csv => FileType::Csv,
            Format::Parquet => FileType::Parquet,
            Format::OnDiskVortex => FileType::Vortex,
            other => vortex_panic!("Format {other} isn't supported on Public BI"),
        };

        runtime.block_on(dataset.register_tables(&session, file_type))?;

        for (query_idx, query) in queries.clone().into_iter() {
            let mut runs = Vec::with_capacity(args.iterations);
            let mut last_plan = None;
            for iteration in 0..args.iterations {
                let exec_duration = runtime.block_on(async {
                    let start = Instant::now();
                    let context = session.clone();
                    let query = query.clone();
                    last_plan = tokio::task::spawn(async move {
                        Some(
                            df::execute_query(&context, &query)
                                .instrument(info_span!("execute_query", query_idx, iteration))
                                .await
                                .unwrap_or_else(|e| {
                                    vortex_panic!("executing query {query_idx}: {e}")
                                })
                                .1,
                        )
                    })
                    .in_current_span()
                    .await
                    .vortex_expect("Failed to spawn query");

                    start.elapsed()
                });
                runs.push(exec_duration);
            }

            let plan = last_plan.vortex_expect("must have at least one iteration");

            metrics.push((
                query_idx,
                format,
                VortexMetricsFinder::find_all(plan.as_ref()),
            ));

            all_measurements.push(QueryMeasurement {
                query_idx,
                target: *target,
                benchmark_dataset: BenchmarkDataset::PublicBi {
                    name: pbi_dataset.name.clone(),
                },
                storage: STORAGE_NVME.to_owned(),
                runs,
            });

            progress_bar.inc(1);
        }
    }

    let mut writer: Box<dyn Write> = if let Some(output_path) = args.output_path {
        Box::new(File::create(output_path)?)
    } else {
        let stdout = stdout();
        Box::new(stdout.lock())
    };

    match args.display_format {
        DisplayFormat::Table => {
            if args.display_metrics {
                for (query, format, metric_sets) in metrics {
                    println!("\nmetrics for query={query}, {format}:");
                    for (idx, metric_set) in metric_sets.into_iter().enumerate() {
                        println!("scan[{idx}]:");
                        for m in metric_set
                            .timestamps_removed()
                            .aggregate()
                            .sorted_for_display()
                            .iter()
                        {
                            println!("{m}");
                        }
                    }
                }
            }
            render_table(&mut writer, all_measurements, &args.targets)
        }
        DisplayFormat::GhJson => print_measurements_json(&mut writer, all_measurements),
    }
}
