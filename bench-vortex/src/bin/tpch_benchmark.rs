use std::cmp::Reverse;
use std::process::ExitCode;
use std::sync;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use bench_vortex::setup_logger;
use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::{
    load_datasets, run_tpch_query, tpch_queries, Format, EXPECTED_ROW_COUNTS,
};
use clap::{ArgAction, Parser};
use futures::future::try_join_all;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::LevelFilter;
use serde::Serialize;
use tabled::builder::Builder;
use tabled::settings::Style;
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
    #[arg(short, long, default_value = "10")]
    iterations: usize,
    #[arg(long)]
    vortex_only: bool,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long)]
    display_format: DisplayFormat,
}

#[derive(clap::ValueEnum, Default, Clone, Debug)]
enum DisplayFormat {
    #[default]
    Table,
    GhJson,
}

struct Measurement {
    query_idx: usize,
    time: Duration,
    format: Format,
}

// impl Tabled for Measurement {
//     const LENGTH: usize = 3;

//     fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
//         vec![]
//     }

//     fn headers() -> Vec<std::borrow::Cow<'static, str>> {
//         vec["query_idx".into(), "time".into(), "format".into()]
//     }
// }

#[derive(Serialize)]
struct JsonValue {
    name: String,
    unit: String,
    value: u128,
}

impl Measurement {
    fn to_json(&self) -> JsonValue {
        let name = format!(
            "{format}/q{query_idx}",
            format = self.format,
            query_idx = self.query_idx
        );

        JsonValue {
            name,
            unit: "ms/iter".to_string(),
            value: self.time.as_millis(),
        }
    }
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
        Some(1) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build(),
        Some(n) => tokio::runtime::Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build(),
    }
    .expect("Failed building the Runtime");

    runtime.block_on(bench_main(
        args.queries,
        args.exclude_queries,
        args.iterations,
        args.warmup,
        args.vortex_only,
        args.display_format,
    ))
}

fn render_table(receiver: Receiver<Measurement>, formats: &[Format]) -> anyhow::Result<()> {
    let mut measurements: HashMap<Format, Vec<Measurement>> = HashMap::default();

    while let Ok(m) = receiver.recv() {
        measurements.entry(m.format).or_default().push(m);
    }

    measurements.values_mut().for_each(|v| {
        v.sort_by_key(|m| Reverse(m.query_idx));
    });

    let mut table_builder = Builder::default();

    let mut header = vec!["Query".to_string()];
    header.extend(formats.iter().map(|f| format!("{:?}", f)));
    table_builder.push_record(header);

    for query_idx in 0_usize..22 {
        let mut row = vec![query_idx.to_string()];
        for format in formats {
            let measurement = &measurements[format][query_idx];
            row.push(measurement.time.as_millis().to_string());
        }
        table_builder.push_record(row);
    }

    let table = table_builder
        .build()
        .with(Style::modern())
        .with(Colorization)
        .to_string();
    println!("{table}");

    Ok(())
}

fn print_measurements_json(receiver: Receiver<Measurement>) -> anyhow::Result<()> {
    let mut measurements = Vec::new();

    while let Ok(m) = receiver.recv() {
        measurements.push(m.to_json());
    }

    let output = serde_json::to_string(&measurements)?;

    println!("{output}");

    Ok(())
}

async fn bench_main(
    queries: Option<Vec<usize>>,
    exclude_queries: Option<Vec<usize>>,
    iterations: usize,
    warmup: bool,
    vortex_only: bool,
    display_format: DisplayFormat,
) -> ExitCode {
    // uncomment the below to enable trace logging of datafusion execution
    // setup_logger(LevelFilter::Trace);

    // Run TPC-H data gen.
    let data_dir = DBGen::new(DBGenOptions::default()).generate().unwrap();

    // The formats to run against (vs the baseline)
    let formats = if vortex_only {
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
            Format::InMemoryVortex {
                enable_pushdown: false,
            },
            Format::InMemoryVortex {
                enable_pushdown: true,
            },
            Format::OnDiskVortex {
                enable_compression: true,
            },
            Format::OnDiskVortex {
                enable_compression: false,
            },
        ]
    };

    // Load datasets
    let ctxs = try_join_all(
        formats
            .iter()
            .map(|format| load_datasets(&data_dir, *format)),
    )
    .await
    .unwrap();

    let query_count = queries.as_ref().map_or(22, |c| c.len());

    // Setup a progress bar
    let progress = ProgressBar::new((query_count * formats.len()) as u64);

    // Send back a channel with the results of Row.
    let (measurements_tx, measurements_rx) = sync::mpsc::channel();
    let (row_count_tx, row_count_rx) = sync::mpsc::channel();

    for (query_idx, sql_queries) in tpch_queries() {
        if queries
            .as_ref()
            .map_or(false, |included| !included.contains(&query_idx))
        {
            continue;
        }

        if exclude_queries
            .as_ref()
            .map_or(false, |e| e.contains(&query_idx))
        {
            continue;
        }
        let ctxs = ctxs.clone();
        let tx = measurements_tx.clone();
        let count_tx = row_count_tx.clone();
        let progress = progress.clone();
        let formats = formats.clone();
        rayon::spawn_fifo(move || {
            // let mut elapsed_us = Vec::new();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            for (ctx, format) in ctxs.iter().zip(formats.iter()) {
                if warmup {
                    for i in 0..3 {
                        let row_count =
                            rt.block_on(run_tpch_query(ctx, &sql_queries, query_idx, *format));
                        if i == 0 {
                            count_tx.send((query_idx, *format, row_count)).unwrap();
                        }
                    }
                }

                let mut measures = Vec::new();
                for _ in 0..iterations {
                    let start = Instant::now();
                    rt.block_on(run_tpch_query(ctx, &sql_queries, query_idx, *format));
                    let elapsed = start.elapsed();
                    measures.push(elapsed);
                }
                let fastest = measures.iter().cloned().min().unwrap();

                tx.send(Measurement {
                    query_idx,
                    time: fastest,
                    format: *format,
                })
                .unwrap();

                progress.inc(1);
            }

            // let baseline = elapsed_us.first().unwrap();
            // // yellow: 10% slower than baseline
            // let yellow = baseline.as_micros() + (baseline.as_micros() / 10);
            // // red: 50% slower than baseline
            // let red = baseline.as_micros() + (baseline.as_micros() / 2);
            // cells.push(Cell::new(&format!("{} us", baseline.as_micros())).style_spec("b"));
            // for measure in elapsed_us.iter().skip(1) {
            //     let style_spec = if measure.as_micros() > red {
            //         "bBr"
            //     } else if measure.as_micros() > yellow {
            //         "bFdBy"
            //     } else {
            //         "bFdBG"
            //     };
            //     cells.push(
            //         Cell::new(&format!(
            //             "{} us ({:.2})",
            //             measure.as_micros(),
            //             measure.as_micros() as f64 / baseline.as_micros() as f64
            //         ))
            //         .style_spec(style_spec),
            //     );
            // }
        });
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
                    println!("Mismatched row count {row_count} instead of {expected_row_count} in query {idx} for format {format:?}");
                    mismatched = true;
                }
            })
    }

    match display_format {
        DisplayFormat::Table => {
            render_table(measurements_rx, &formats).unwrap();
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements_rx).unwrap();
        }
    }

    if mismatched {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
