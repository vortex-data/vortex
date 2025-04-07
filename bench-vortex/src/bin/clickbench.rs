use std::fs::{self};
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{self, Flavor, HITS_SCHEMA, clickbench_queries};
use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::{
    Format, IdempotentPath as _, default_env_filter, execute_physical_plan,
    feature_flagged_allocator, get_session_with_cache, make_object_store, physical_plan,
};
use clap::Parser;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::warn;
use tokio::runtime::Builder;
use tracing::info_span;
use tracing_futures::Instrument;
use url::Url;
use vortex::error::{VortexExpect, vortex_panic};
use vortex_datafusion::persistent::metrics::VortexMetricsFinder;

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Format::Parquet, Format::OnDiskVortex])]
    formats: Vec<Format>,
    #[arg(long)]
    only_vortex: bool,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    #[arg(long, default_value_t = false)]
    emulate_object_store: bool,
    #[arg(long)]
    export_spans: bool,
    #[arg(long, value_enum, default_value_t = Flavor::Partitioned)]
    flavor: Flavor,
    #[arg(long)]
    use_remote_data_dir: Option<String>,
    #[arg(long, default_value_t = false)]
    single_file: bool,
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
            .file("clickbench.trace.json")
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

    if args.only_vortex {
        panic!("use `--formats vortex` instead of `--only-vortex`");
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
            let basepath = format!("clickbench_{}", args.flavor).to_data_path();
            let client = reqwest::blocking::Client::default();

            args.flavor.download(&client, basepath.as_path())?;
            Url::parse(
                ("file:".to_owned() + basepath.to_str().vortex_expect("path should be utf8") + "/")
                    .as_ref(),
            )
            .unwrap()
        }
        Some(remote_data_dir) => {
            // e.g. "s3://vortex-bench-dev-eu/parquet/"
            if !remote_data_dir.ends_with("/") {
                log::warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            log::info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\n",
                    "--use-remote-data-dir) and upload data/clickbench/ to some remote location.",
                ),
                remote_data_dir,
            );
            Url::parse(&remote_data_dir).unwrap()
        }
    };

    let queries = match args.queries.clone() {
        None => clickbench_queries(),
        Some(queries) => clickbench_queries()
            .into_iter()
            .filter(|(q_idx, _)| queries.iter().contains(q_idx))
            .collect(),
    };

    let progress_bar = ProgressBar::new((queries.len() * args.formats.len()) as u64);

    let mut all_measurements = Vec::default();

    let mut metrics = Vec::new();
    for format in &args.formats {
        let session_context = get_session_with_cache(args.emulate_object_store);
        // register object store to the session
        let _ = make_object_store(&session_context, &url)?;
        let context = session_context.clone();
        let mut plans = Vec::new();

        match format {
            Format::Parquet => runtime.block_on(async {
                clickbench::register_parquet_files(
                    &context,
                    "hits",
                    &url,
                    &HITS_SCHEMA,
                    args.single_file,
                )
                .await
                .unwrap()
            }),
            Format::OnDiskVortex => {
                runtime.block_on(async {
                    if url.scheme() == "file" {
                        clickbench::convert_parquet_to_vortex(
                            context.clone(),
                            &url.to_file_path().unwrap(),
                        )
                        .await
                        .unwrap();
                    }
                    clickbench::register_vortex_files(
                        context.clone(),
                        "hits",
                        &url,
                        &HITS_SCHEMA,
                        args.single_file,
                    )
                    .await
                    .unwrap();
                });
            }
            other => vortex_panic!("Format {other} isn't supported on ClickBench"),
        }

        for (query_idx, query) in queries.clone().into_iter() {
            let mut fastest_result = Duration::from_millis(u64::MAX);
            let mut last_plan = None;
            for iteration in 0..args.iterations {
                let exec_duration = runtime.block_on(async {
                    let start = Instant::now();
                    let context = context.clone();
                    let query = query.clone();
                    last_plan = tokio::task::spawn(async move {
                        let execution_plan = physical_plan(&context, &query)
                            .instrument(info_span!("create_physical_plan", query_idx, iteration))
                            .await
                            .unwrap_or_else(|e| panic!("physical plan {query_idx}: {e}"));

                        execute_physical_plan(&context, execution_plan.clone())
                            .instrument(info_span!("execute_query", query_idx, iteration))
                            .await
                            .unwrap_or_else(|e| panic!("executing query {query_idx}: {e}"));
                        Some(execution_plan.clone())
                    })
                    .await
                    .unwrap();

                    start.elapsed()
                });

                fastest_result = fastest_result.min(exec_duration);
            }

            if let Some(plan) = last_plan.clone() {
                plans.push((query_idx, plan));
            }
            progress_bar.inc(1);

            let plan = last_plan.expect("must have at least one iteration");
            if args.emit_plan {
                fs::write(
                    format!("clickbench_{format}_q{query_idx:02}.plan",),
                    format!("{:#?}", plan),
                )
                .expect("Unable to write file");

                fs::write(
                    format!("clickbench_{format}_q{query_idx:02}.short.plan",),
                    format!(
                        "{}",
                        DisplayableExecutionPlan::with_full_metrics(plan.as_ref())
                            .set_show_schema(true)
                            .set_show_statistics(true)
                            .indent(true)
                    ),
                )
                .expect("Unable to write file");
            }
            metrics.push((
                query_idx,
                format,
                VortexMetricsFinder::find_all(plan.as_ref()),
            ));
            all_measurements.push(QueryMeasurement {
                query_idx,
                storage: "nvme".to_string(),
                time: fastest_result,
                format: *format,
                dataset: "clickbench".to_string(),
            });
        }
        if args.export_spans {
            if let Err(e) = runtime.block_on(async move { export_plan_spans(*format, plans).await })
            {
                warn!("failed to export spans {e}");
            }
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
            render_table(all_measurements, &args.formats, RatioMode::Time).unwrap()
        }
        DisplayFormat::GhJson => print_measurements_json(all_measurements).unwrap(),
    }

    Ok(())
}
