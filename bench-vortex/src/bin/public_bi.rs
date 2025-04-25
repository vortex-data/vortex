use std::time::{Duration, Instant};

use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::MetricsSetExt;
use bench_vortex::public_bi::{FileType, PBI_DATASETS, PBIDataset};
use bench_vortex::utils::constants::STORAGE_NVME;
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{Engine, Format, Target, default_env_filter, df};
use clap::Parser;
use indicatif::ProgressBar;
use itertools::Itertools;
use tracing::info_span;
use tracing_futures::Instrument;
use vortex::error::vortex_panic;
use vortex_datafusion::persistent::metrics::VortexMetricsFinder;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Format::Parquet, Format::OnDiskVortex])]
    formats: Vec<Format>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    emulate_object_store: bool,
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,
    #[arg(short, long, value_delimiter = ',')]
    dataset: PBIDataset,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Capture `RUST_LOG` configuration
    let filter = default_env_filter(args.verbose);

    #[cfg(not(feature = "tracing"))]
    bench_vortex::setup_logger(filter);

    // We need the guard to live to the end of the function, so can't create it in the if-block
    #[cfg(feature = "tracing")]
    let _trace_guard = {
        use std::io::IsTerminal;

        use tracing_subscriber::prelude::*;

        let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
            .include_args(true)
            .trace_style(tracing_chrome::TraceStyle::Async)
            .file("public_bi.trace.json")
            .build();

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_level(true)
            .with_line_number(true)
            .with_ansi(std::io::stderr().is_terminal());

        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .with(fmt_layer)
            .init();
        _guard
    };

    let runtime = new_tokio_runtime(args.threads);

    let targets = args
        .formats
        .iter()
        .map(|f| Target::new(Engine::DataFusion, *f))
        .collect_vec();

    let pbi_dataset = PBI_DATASETS.get(args.dataset);
    let queries = match args.queries.clone() {
        None => pbi_dataset.queries()?,
        Some(queries) => pbi_dataset
            .queries()?
            .into_iter()
            .filter(|(q_idx, _)| queries.iter().contains(q_idx))
            .collect(),
    };

    let progress_bar = ProgressBar::new((queries.len() * args.formats.len()) as u64);
    let mut all_measurements = Vec::default();
    let mut metrics = Vec::new();

    let dataset = pbi_dataset.dataset().expect("failed to parse data urls");
    tracing::info!("preparing files");
    // download csvs, unzip, convert to parquet, and convert that to vortex
    runtime.block_on(dataset.write_as_vortex());

    for target in &targets {
        let format = target.format();
        let session =
            df::get_session_context(args.emulate_object_store, args.disable_datafusion_cache);

        let file_type = match format {
            Format::Csv => FileType::Csv,
            Format::Parquet => FileType::Parquet,
            Format::OnDiskVortex => FileType::Vortex,
            other => vortex_panic!("Format {other} isn't supported on Public BI"),
        };

        runtime
            .block_on(dataset.register_tables(&session, file_type))
            .expect("failed to register");

        for (query_idx, query) in queries.clone().into_iter() {
            let mut fastest_run = Duration::from_millis(u64::MAX);
            let mut last_plan = None;
            for iteration in 0..args.iterations {
                let exec_duration = runtime.block_on(async {
                    let start = Instant::now();
                    let context = session.clone();
                    let query = query.clone();
                    last_plan = tokio::task::spawn(async move {
                        let plan = df::execute_query(&context, &query)
                            .instrument(info_span!("execute_query", query_idx, iteration))
                            .await
                            .unwrap_or_else(|e| panic!("executing query {query_idx}: {e}"))
                            .1;
                        Some(plan.clone())
                    })
                    .await
                    .unwrap();

                    start.elapsed()
                });
                fastest_run = fastest_run.min(exec_duration);
            }

            let plan = last_plan.expect("must have at least one iteration");

            metrics.push((
                query_idx,
                format,
                VortexMetricsFinder::find_all(plan.as_ref()),
            ));

            all_measurements.push(QueryMeasurement {
                query_idx,
                target: *target,
                storage: STORAGE_NVME.to_owned(),
                fastest_run,
                dataset: pbi_dataset.name.to_owned(),
            });

            progress_bar.inc(1);
        }
    }

    match args.display_format {
        DisplayFormat::Table => {
            for (query, format, metric_sets) in metrics {
                println!();
                println!("metrics for query={query}, {format}:");
                for (idx, metric_set) in metric_sets.into_iter().enumerate() {
                    println!("scan[{idx}]:");
                    for m in metric_set
                        .timestamps_removed()
                        .aggregate()
                        .sorted_for_display()
                        .iter()
                    {
                        println!("{}", m);
                    }
                }
            }
            render_table(all_measurements, RatioMode::Time, &targets).unwrap()
        }
        DisplayFormat::GhJson => print_measurements_json(all_measurements).unwrap(),
    }

    Ok(())
}
