#![feature(exit_status_error)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{self, clickbench_queries, HITS_SCHEMA};
use bench_vortex::display::{print_measurements_json, render_table, DisplayFormat};
use bench_vortex::{
    execute_query, idempotent, physical_plan, setup_logger, Format, IdempotentPath as _,
    Measurement,
};
use clap::Parser;
use datafusion::prelude::SessionContext;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::LevelFilter;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};
use tokio::runtime::Builder;
use vortex::error::vortex_panic;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "8")]
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
}

fn main() {
    let args = Args::parse();

    setup_logger(if args.verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });

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

    // The clickbench-provided file is missing some higher-level type info, so we reprocess it
    // to add that info, see https://github.com/ClickHouse/ClickBench/issues/7.
    (0_u32..100).into_par_iter().for_each(|idx| {
        let output_path = basepath.join(format!("hits_{idx}.parquet"));
        idempotent(&output_path, |output_path| {
            eprintln!("Fixing parquet file {idx}");

            // We need to set the home directory because GitHub Actions doesn't set it in a way
            // that DuckDB respects.
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ci-runner".to_string());

            let command = format!(
                "
                SET home_directory='{home}';
                INSTALL HTTPFS;
                COPY (SELECT * REPLACE
                    (epoch_ms(EventTime * 1000) AS EventTime, \
                    epoch_ms(ClientEventTime * 1000) AS ClientEventTime, \
                    epoch_ms(LocalEventTime * 1000) AS LocalEventTime, \
                        DATE '1970-01-01' + INTERVAL (EventDate) DAYS AS EventDate) \
                FROM read_parquet('https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{idx}.parquet', binary_as_string=True)) TO '{}' (FORMAT 'parquet');",
                output_path.to_str().unwrap()
            );
            Command::new("duckdb")
                .arg("-c")
                .arg(command)
                .status()?
                .exit_ok()?;

            anyhow::Ok(PathBuf::from(output_path))
        })
        .unwrap();
    });

    let formats = if args.only_vortex {
        vec![Format::OnDiskVortex {
            enable_compression: true,
        }]
    } else {
        vec![
            Format::Parquet,
            Format::OnDiskVortex {
                enable_compression: true,
            },
        ]
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
        let session_context = SessionContext::new();
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
            Format::OnDiskVortex {
                enable_compression: true,
            } => {
                runtime.block_on(async {
                    clickbench::register_vortex_files(
                        &context,
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
            }

            let mut fastest_result = Duration::from_millis(u64::MAX);
            for _ in 0..args.iterations {
                let exec_duration = runtime.block_on(async {
                    let start = Instant::now();
                    execute_query(&context, &query).await.unwrap();
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
