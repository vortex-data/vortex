use std::process::ExitCode;
use std::time::{Duration, Instant};

use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::duckdb::{DuckdbTpchOptions, generate_tpch};
use bench_vortex::tpch::{
    EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, TPC_H_ROW_COUNT_ARRAY_LENGTH, load_datasets,
    run_tpch_query, tpch_queries,
};
use bench_vortex::{Format, default_env_filter, feature_flagged_allocator};
use clap::{Parser, ValueEnum};
use datafusion_physical_plan::metrics::{Label, MetricsSet};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{info, warn};
use tokio::runtime::Builder;
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::error::VortexExpect;
use vortex_datafusion::persistent::metrics::VortexMetricsFinder;

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long)]
    use_remote_data_dir: Option<String>,
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Format::Arrow, Format::Parquet, Format::OnDiskVortex])]
    formats: Vec<Format>,
    #[arg(long, default_value_t = 1)]
    scale_factor: u8,
    #[arg(long)]
    only_vortex: bool,
    #[arg(short)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    emulate_object_store: bool,
    #[arg(long, default_value_t, value_enum)]
    data_generator: DataGenerator,
    #[arg(long)]
    all_metrics: bool,
    #[arg(long)]
    export_spans: bool,
}

#[derive(ValueEnum, Default, Clone, Debug)]
pub enum DataGenerator {
    #[default]
    Dbgen,
    Duckdb,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let filter = default_env_filter(args.verbose);
    #[cfg(not(feature = "tracing"))]
    bench_vortex::setup_logger(filter);

    // We need the guard to live to the end of the function, so can't create it in the if-block
    #[cfg(feature = "tracing")]
    let _trace_guard = {
        use tracing_subscriber::prelude::*;

        let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
            .include_args(true)
            .trace_style(tracing_chrome::TraceStyle::Async)
            .file("tpch.trace.json")
            .build();

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_level(true)
            .with_line_number(true);

        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .with(fmt_layer)
            .init();
        _guard
    };

    if args.only_vortex {
        panic!("use `--formats vortex,arrow` instead of `--only-vortex`");
    }

    let runtime = match args.threads {
        Some(0) => panic!("Can't use 0 threads for runtime"),
        Some(1) => Builder::new_current_thread().enable_all().build(),
        Some(n) => Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => Builder::new_multi_thread().enable_all().build(),
    }
    .expect("Failed building the Runtime");

    let url = match args.use_remote_data_dir {
        None => {
            let data_dir = match args.data_generator {
                DataGenerator::Duckdb => {
                    generate_tpch(DuckdbTpchOptions::default().with_scale_factor(args.scale_factor))
                        .unwrap()
                }
                DataGenerator::Dbgen => {
                    let db_gen_options =
                        DBGenOptions::default().with_scale_factor(args.scale_factor);
                    DBGen::new(db_gen_options).generate().unwrap()
                }
            };

            info!(
                "Using existing or generating new files located at {}.",
                data_dir.display()
            );
            Url::parse(
                ("file:".to_owned() + data_dir.to_str().vortex_expect("path should be utf8") + "/")
                    .as_ref(),
            )
            .unwrap()
        }
        Some(tpch_benchmark_remote_data_dir) => {
            // e.g. "s3://vortex-bench-dev-eu/parquet/"
            //
            // The trailing slash is significant!
            //
            // The folder must already be populated with data!
            if !tpch_benchmark_remote_data_dir.ends_with("/") {
                warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\n",
                    "--use-remote-data-dir) and upload data/tpch/1/ to some remote location.",
                ),
                tpch_benchmark_remote_data_dir,
            );
            Url::parse(&tpch_benchmark_remote_data_dir).unwrap()
        }
    };

    runtime.block_on(bench_main(
        args.queries,
        args.exclude_queries,
        args.iterations,
        args.formats,
        args.display_format,
        args.emulate_object_store,
        args.scale_factor,
        url,
        args.all_metrics,
        args.export_spans,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn bench_main(
    queries: Option<Vec<usize>>,
    exclude_queries: Option<Vec<usize>>,
    iterations: usize,
    formats: Vec<Format>,
    display_format: DisplayFormat,
    emulate_object_store: bool,
    scale_factor: u8,
    url: Url,
    display_all_metrics: bool,
    export_spans: bool,
) -> ExitCode {
    let expected_row_counts = if scale_factor == 1 {
        EXPECTED_ROW_COUNTS_SF1
    } else if scale_factor == 10 {
        EXPECTED_ROW_COUNTS_SF10
    } else {
        panic!(
            "Scale factor {} not supported due to lack of expected row counts.",
            scale_factor
        );
    };

    info!(
        "Benchmarking against these formats: {}.",
        formats.iter().join(", ")
    );

    let query_count = queries.as_ref().map_or(22, |c| c.len());

    // Set up a progress bar
    let progress = ProgressBar::new((query_count * formats.len()) as u64);

    let mut row_counts = Vec::new();
    let mut measurements = Vec::new();

    let mut metrics = MetricsSet::new();

    for format in formats.iter().copied() {
        let mut plans = Vec::new();
        // Load datasets
        let ctx = load_datasets(&url, format, emulate_object_store)
            .await
            .unwrap();

        for (query_idx, sql_queries) in tpch_queries() {
            if queries
                .as_ref()
                .is_some_and(|included| !included.contains(&query_idx))
            {
                continue;
            }

            if exclude_queries
                .as_ref()
                .is_some_and(|excluded| excluded.contains(&query_idx))
            {
                continue;
            }

            let mut fastest_result = Duration::from_millis(u64::MAX);
            for i in 0..iterations {
                let start = Instant::now();
                let (row_count, plan) = run_tpch_query(&ctx, &sql_queries, query_idx).await;
                let elapsed = start.elapsed();
                fastest_result = fastest_result.min(elapsed);

                if i == 0 {
                    row_counts.push((query_idx, format, row_count));
                    // gather metrics
                    for (idx, m) in VortexMetricsFinder::find_all(plan.as_ref())
                        .into_iter()
                        .enumerate()
                    {
                        metrics.merge_all_with_label(
                            m,
                            &[
                                Label::new("query_idx", query_idx.to_string()),
                                Label::new("vortex_exec_idx", idx.to_string()),
                            ],
                        );
                    }
                    plans.push((query_idx, plan));
                }
            }

            let storage = match url.scheme() {
                "s3" => "s3",
                "gcs" => "gcs",
                "file" => "nvme",
                otherwise => {
                    warn!("unknown URL scheme: {}", otherwise);
                    return ExitCode::FAILURE;
                }
            }
            .to_owned();

            measurements.push(QueryMeasurement {
                query_idx,
                storage,
                time: fastest_result,
                format,
                dataset: "tpch".to_string(),
            });

            progress.inc(1);
        }
        if export_spans {
            if let Err(e) = export_plan_spans(format, plans).await {
                warn!("failed to export spans {e}");
            }
        }
    }

    let mut format_row_counts: HashMap<Format, Vec<usize>> = HashMap::new();
    for (idx, format, row_count) in row_counts {
        format_row_counts
            .entry(format)
            .or_insert_with(|| vec![0; TPC_H_ROW_COUNT_ARRAY_LENGTH])[idx] = row_count;
    }

    progress.finish();

    let mut mismatched = false;
    for (format, row_counts) in format_row_counts {
        row_counts
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| queries.as_ref().map(|q| q.contains(idx)).unwrap_or(true))
            .filter(|(idx, _)| exclude_queries.as_ref().map(|excluded| !excluded.contains(idx)).unwrap_or(true))
            .for_each(|(idx, actual_row_count)| {
                let expected_row_count = expected_row_counts[idx];
                if actual_row_count != expected_row_count {
                    if idx == 15 && actual_row_count == 0 {
                        warn!(
                            "*IGNORING* mismatched row count {} instead of {} for format {:?} because Query 15 is flaky. See: https://github.com/spiraldb/vortex/issues/2395",
                            actual_row_count,
                            expected_row_count,
                            format,
                        );
                    } else  {
                        warn!(
                            "Mismatched row count {} instead of {} in query {} for format {:?}",
                            actual_row_count,
                            expected_row_count,
                            idx,
                            format,
                        );
                        mismatched = true;
                    }
                }
            })
    }

    match display_format {
        DisplayFormat::Table => {
            if !display_all_metrics {
                metrics = metrics.aggregate();
            }
            for m in metrics.timestamps_removed().sorted_for_display().iter() {
                println!("{}", m);
            }
            render_table(measurements, &formats, RatioMode::Time).unwrap();
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements).unwrap();
        }
    }

    if mismatched {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
