use std::fs::{self};
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{self, Flavor, HITS_SCHEMA, clickbench_queries};
use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::export_plan_spans;
use bench_vortex::{
    Format, IdempotentPath as _, default_env_filter, execute_physical_plan,
    feature_flagged_allocator, get_session_with_cache, physical_plan,
};
use clap::Parser;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::warn;
use tokio::runtime::Builder;
use tracing::info_span;
use tracing_futures::Instrument;
use vortex::error::vortex_panic;

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "5")]
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
    #[arg(long, default_value = "false")]
    emit_plan: bool,
    #[arg(long, default_value = "false")]
    emulate_object_store: bool,
    #[arg(long)]
    export_spans: bool,
    #[arg(long, value_enum, default_value_t = Flavor::Partitioned)]
    flavor: Flavor,
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
        use tracing_subscriber::prelude::*;

        let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
            .include_args(true)
            .trace_style(tracing_chrome::TraceStyle::Async)
            .file("clickbench.trace.json")
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
    let basepath = format!("clickbench_{}", args.flavor).to_data_path();
    let client = reqwest::blocking::Client::default();

    args.flavor.download(&client, basepath.as_path())?;

    let queries = match args.queries.clone() {
        None => clickbench_queries(),
        Some(queries) => clickbench_queries()
            .into_iter()
            .filter(|(q_idx, _)| queries.iter().contains(q_idx))
            .collect(),
    };

    let progress_bar = ProgressBar::new((queries.len() * args.formats.len()) as u64);

    let mut all_measurements = Vec::default();

    for format in &args.formats {
        let session_context = get_session_with_cache(args.emulate_object_store);
        let context = session_context.clone();
        let mut plans = Vec::new();

        match format {
            Format::Parquet => runtime.block_on(async {
                clickbench::register_parquet_files(
                    &context,
                    "hits",
                    basepath.as_path(),
                    &HITS_SCHEMA,
                )
                .await
                .unwrap()
            }),
            Format::OnDiskVortex => {
                runtime.block_on(async {
                    clickbench::register_vortex_files(
                        context.clone(),
                        "hits",
                        basepath.as_path(),
                        &HITS_SCHEMA,
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

            if args.emit_plan {
                let plan = last_plan.expect("must have at least one iteration");
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
            render_table(all_measurements, &args.formats, RatioMode::Time).unwrap()
        }
        DisplayFormat::GhJson => print_measurements_json(all_measurements).unwrap(),
    }

    Ok(())
}
