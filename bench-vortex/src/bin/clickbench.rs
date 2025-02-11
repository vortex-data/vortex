#![feature(exit_status_error)]

use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{self, clickbench_queries, HITS_SCHEMA};
use bench_vortex::display::{print_measurements_json, render_table, DisplayFormat};
use bench_vortex::{
    default_env_filter, execute_query, feature_flagged_allocator, get_session_with_cache,
    idempotent, physical_plan, Format, IdempotentPath as _, Measurement,
};
use clap::Parser;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use indicatif::ProgressBar;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};
use tokio::runtime::Builder;
use tracing::info_span;
use tracing_futures::Instrument;
use vortex::error::{vortex_panic, VortexExpect};

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "5")]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
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
}

fn main() {
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
            .file("clickbench.trace.json")
            .build();

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_file(true)
            .with_level(true)
            .with_line_number(true);

        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .with(fmt_layer)
            .init();
        _guard
    };

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
    let basepath = "clickbench".to_data_path();
    let client = reqwest::blocking::Client::default();

    // The clickbench-provided file is missing some higher-level type info, so we reprocess it
    // to add that info, see https://github.com/ClickHouse/ClickBench/issues/7.
    (0_u32..100).into_par_iter().for_each(|idx| {
        let output_path = basepath.join("parquet").join(format!("hits_{idx}.parquet"));
        idempotent(&output_path, |output_path| {
            eprintln!("Downloading file {idx}");
            let url = format!("https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_{idx}.parquet");


            let make_req = || client.get(&url).send();
            let mut output = None;

            for attempt in 1..4 {
                match make_req() {
                    Ok(r) => {
                          output = Some(r.error_for_status());
                          break;
                    },
                    Err(e) => {
                        eprintln!("Request for file {idx} timed out, retying for the {attempt} time");
                        output = Some(Err(e));
                    }
                }

                // Very basic backoff mechanism
                std::thread::sleep(Duration::from_secs(attempt));
            }

            let mut response = output.vortex_expect("Must have value here")?;
            let mut file = File::create(output_path)?;
            response.copy_to(&mut file)?;

            anyhow::Ok(PathBuf::from(output_path))
        })
        .unwrap();
    });

    let formats = if args.only_vortex {
        vec![Format::OnDiskVortex]
    } else {
        vec![Format::Parquet, Format::OnDiskVortex]
    };

    let queries = match args.queries.clone() {
        None => clickbench_queries(),
        Some(queries) => clickbench_queries()
            .into_iter()
            .filter(|(q_idx, _)| queries.iter().contains(q_idx))
            .collect(),
    };

    let progress_bar = ProgressBar::new((queries.len() * formats.len()) as u64);

    let mut all_measurements = Vec::default();

    for format in &formats {
        let session_context = get_session_with_cache(args.emulate_object_store);
        let context = session_context.clone();
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
            if args.emit_plan {
                let plan = runtime.block_on(physical_plan(&context, &query)).unwrap();
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

            let mut fastest_result = Duration::from_millis(u64::MAX);
            for iteration in 0..args.iterations {
                let exec_duration = runtime.block_on(async {
                    let start = Instant::now();
                    let context = context.clone();
                    let query = query.clone();
                    tokio::task::spawn(async move {
                        execute_query(&context, &query)
                            .instrument(info_span!("execute_query", query_idx, iteration))
                            .await
                            .unwrap_or_else(|e| panic!("executing query {query_idx}: {e}"));
                    })
                    .await
                    .unwrap();

                    start.elapsed()
                });

                fastest_result = fastest_result.min(exec_duration);
            }

            progress_bar.inc(1);

            all_measurements.push(Measurement {
                query_idx,
                time: fastest_result,
                format: *format,
                dataset: "clickbench".to_string(),
            });
        }
    }

    match args.display_format {
        DisplayFormat::Table => render_table(all_measurements, &formats).unwrap(),
        DisplayFormat::GhJson => print_measurements_json(all_measurements).unwrap(),
    }
}
