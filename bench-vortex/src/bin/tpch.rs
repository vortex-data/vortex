use std::process::ExitCode;
use std::time::{Duration, Instant};

use bench_vortex::display::{print_measurements_json, render_table, DisplayFormat};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::tpch::dbgen::{DBGen, DBGenOptions};
use bench_vortex::tpch::{load_datasets, run_tpch_query, tpch_queries, EXPECTED_ROW_COUNTS};
use bench_vortex::{default_env_filter, feature_flagged_allocator, setup_logger, Format};
use clap::Parser;
use futures::future::try_join_all;
use indicatif::ProgressBar;
use itertools::Itertools;
use tokio::runtime::Builder;
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::error::VortexExpect as _;

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
    #[arg(short, long, default_value = "5")]
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
    #[arg(long, default_value = "false")]
    emulate_object_store: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let filter = default_env_filter(args.verbose);
    setup_logger(filter);

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
            let db_gen_options = DBGenOptions::default().with_scale_factor(args.scale_factor);
            let data_dir = DBGen::new(db_gen_options).generate().unwrap();
            eprintln!(
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
            // e.g. "s3://vortex-bench-dev/parquet/"
            //
            // The trailing slash is significant!
            //
            // The folder must already be populated with data!
            if !tpch_benchmark_remote_data_dir.ends_with("/") {
                eprintln!("Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev/parquet/");
            }
            eprintln!(
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
        url,
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
    url: Url,
) -> ExitCode {
    eprintln!(
        "Benchmarking against these formats: {}.",
        formats.iter().join(", ")
    );

    // Load datasets
    let ctxs = try_join_all(
        formats
            .iter()
            .map(|format| load_datasets(&url, *format, emulate_object_store)),
    )
    .await
    .unwrap();

    let query_count = queries.as_ref().map_or(22, |c| c.len());

    // Set up a progress bar
    let progress = ProgressBar::new((query_count * formats.len()) as u64);

    let mut row_counts = Vec::new();
    let mut measurements = Vec::new();

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

        for (ctx, format) in ctxs.iter().zip(formats.iter()) {
            for i in 0..2 {
                let row_count = run_tpch_query(ctx, &sql_queries, query_idx, *format).await;
                if i == 0 {
                    row_counts.push((query_idx, *format, row_count));
                }
            }

            let mut fastest_result = Duration::from_millis(u64::MAX);
            for _ in 0..iterations {
                let start = Instant::now();
                run_tpch_query(ctx, &sql_queries, query_idx, *format).await;
                let elapsed = start.elapsed();
                fastest_result = fastest_result.min(elapsed);
            }

            measurements.push(QueryMeasurement {
                query_idx,
                time: fastest_result,
                format: *format,
                dataset: "tpch".to_string(),
            });

            progress.inc(1);
        }
    }

    let mut format_row_counts: HashMap<Format, Vec<usize>> = HashMap::new();
    for (idx, format, row_count) in row_counts {
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

    match display_format {
        DisplayFormat::Table => {
            render_table(measurements, &formats).unwrap();
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
