use std::process::ExitCode;
use std::sync;
use std::time::Instant;

use bench_vortex::display::{print_measurements_json, render_table, DisplayFormat};
use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::{load_datasets, run_tpch_query, tpch_queries, EXPECTED_ROW_COUNTS};
use bench_vortex::{setup_logger, Format, Measurement};
use clap::{ArgAction, Parser};
use futures::future::try_join_all;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::LevelFilter;
use tokio::runtime::Builder;
use vortex::aliases::hash_map::HashMap;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(short, long, default_value_t = true, default_missing_value = "true", action = ArgAction::Set)]
    warmup: bool,
    #[arg(short, long, default_value = "5")]
    iterations: usize,
    #[arg(long)]
    only_vortex: bool,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value = "false")]
    emulate_object_store: bool,
}

fn main() -> ExitCode {
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

    runtime.block_on(bench_main(
        args.queries,
        args.exclude_queries,
        args.iterations,
        args.warmup,
        args.only_vortex,
        args.display_format,
        args.emulate_object_store,
    ))
}

async fn bench_main(
    queries: Option<Vec<usize>>,
    exclude_queries: Option<Vec<usize>>,
    iterations: usize,
    warmup: bool,
    only_vortex: bool,
    display_format: DisplayFormat,
    emulate_object_store: bool,
) -> ExitCode {
    // uncomment the below to enable trace logging of datafusion execution
    // setup_logger(LevelFilter::Trace);

    // Run TPC-H data gen.
    let data_dir = DBGen::new(DBGenOptions::default()).generate().unwrap();

    // The formats to run against (vs the baseline)
    let formats = if only_vortex {
        vec![
            Format::Arrow,
            Format::OnDiskVortex {
                enable_compression: true,
            },
        ]
    } else {
        vec![
            Format::Arrow,
            Format::Parquet,
            Format::OnDiskVortex {
                enable_compression: true,
            },
        ]
    };

    // Load datasets
    let ctxs = try_join_all(
        formats
            .iter()
            .map(|format| load_datasets(&data_dir, *format, emulate_object_store)),
    )
    .await
    .unwrap();

    let query_count = queries.as_ref().map_or(22, |c| c.len());

    // Set up a progress bar
    let progress = ProgressBar::new((query_count * formats.len()) as u64);

    // Send back a channel with the results of Row.
    let (measurements_tx, measurements_rx) = sync::mpsc::channel();
    let (row_count_tx, row_count_rx) = sync::mpsc::channel();

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
        let ctxs = ctxs.clone();
        let tx = measurements_tx.clone();
        let count_tx = row_count_tx.clone();
        let progress = progress.clone();
        let formats = formats.clone();

        for (ctx, format) in ctxs.iter().zip(formats.iter()) {
            if warmup {
                for i in 0..2 {
                    let row_count = run_tpch_query(ctx, &sql_queries, query_idx, *format).await;
                    if i == 0 {
                        count_tx.send((query_idx, *format, row_count)).unwrap();
                    }
                }
            }

            let mut measures = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                run_tpch_query(ctx, &sql_queries, query_idx, *format).await;
                let elapsed = start.elapsed();
                measures.push(elapsed);
            }
            let fastest = measures.iter().cloned().min().unwrap();

            tx.send(Measurement {
                query_idx,
                time: fastest,
                format: *format,
                dataset: "tpch".to_string(),
            })
            .unwrap();

            progress.inc(1);
        }
    }

    // delete parent handle to tx
    drop(measurements_tx);
    drop(row_count_tx);

    let mut format_row_counts: HashMap<Format, Vec<usize>> = HashMap::new();
    while let Ok((idx, format, row_count)) = row_count_rx.recv() {
        format_row_counts
            .entry(format)
            .or_insert_with(|| vec![0; EXPECTED_ROW_COUNTS.len()])[idx] = row_count;
    }

    progress.finish();

    let mut mismatched = false;
    for (format, row_counts) in format_row_counts {
        row_counts
            .into_iter()
            .zip_eq(EXPECTED_ROW_COUNTS)
            .enumerate()
            .filter(|(idx, _)| queries.as_ref().map(|q| q.contains(idx)).unwrap_or(true))
            .for_each(|(idx, (row_count, expected_row_count))| {
                if row_count != expected_row_count {
                    eprintln!("Mismatched row count {row_count} instead of {expected_row_count} in query {idx} for format {format:?}");
                    mismatched = true;
                }
            })
    }

    let all_measurements = measurements_rx.into_iter().collect::<Vec<_>>();

    match display_format {
        DisplayFormat::Table => {
            render_table(all_measurements, &formats).unwrap();
        }
        DisplayFormat::GhJson => {
            print_measurements_json(all_measurements).unwrap();
        }
    }

    if mismatched {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
