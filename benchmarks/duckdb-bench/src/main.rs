// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod validation;

use std::path::PathBuf;

use clap::Parser;
use clap::value_parser;
use duckdb_bench::DuckClient;
use tokio::runtime::Runtime;
use vortex::metrics::tracing::set_global_labels;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::conversions::convert_parquet_directory_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::runner::BenchmarkMode;
use vortex_bench::runner::SqlBenchmarkRunner;
use vortex_bench::runner::filter_queries;
use vortex_bench::setup_logging_and_tracing;

/// Common arguments shared across benchmarks
#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,

    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    tracing: bool,

    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(long, default_value_t = false)]
    delete_duckdb_database: bool,

    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,

    #[arg(short)]
    output_path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long, default_value = "unknown")]
    runner: String,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format))]
    formats: Vec<Format>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,

    /// Print EXPLAIN output for each query instead of running benchmarks.
    #[arg(long, default_value_t = false)]
    explain: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Whether to reuse the DuckDB connection across iterations. Helpful when profiling \
        to keep all work on the same threads"
    )]
    reuse: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let opts = Opts::from(args.options);

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        args.queries.as_ref(),
        args.exclude_queries.as_ref(),
    );

    if args.formats.is_empty() {
        anyhow::bail!("provide a format with --formats");
    }

    // Generate Vortex files from Parquet for any Vortex formats requested
    if benchmark.data_url().scheme() == "file" {
        // This is ugly, but otherwise some complicated async interaction might result in a deadlock
        let runtime = Runtime::new()?;

        runtime.block_on(async {
            benchmark.generate_base_data().await?;

            let base_path = benchmark
                .data_url()
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

            for format in args.formats.iter().copied() {
                match format {
                    Format::OnDiskVortex => {
                        convert_parquet_directory_to_vortex(
                            &base_path,
                            CompactionStrategy::Default,
                        )
                        .await?;
                    }
                    Format::VortexCompact => {
                        convert_parquet_directory_to_vortex(
                            &base_path,
                            CompactionStrategy::Compact,
                        )
                        .await?;
                    }
                    // OnDiskDuckDB tables are created during register_tables by loading from Parquet
                    _ => {}
                }
            }

            anyhow::Ok(())
        })?;
    }

    let mut runner = SqlBenchmarkRunner::new(
        &*benchmark,
        Engine::DuckDB,
        args.runner.clone(),
        args.formats.clone(),
        args.track_memory,
        args.hide_progress_bar,
    )?;

    let benchmark_name = benchmark.dataset().to_string();

    let mode = if args.explain {
        BenchmarkMode::Explain
    } else {
        BenchmarkMode::Run {
            iterations: args.iterations,
        }
    };

    runner.run_all(
        &filtered_queries,
        mode,
        |format| {
            let ctx = DuckClient::new(
                &*benchmark,
                format,
                args.delete_duckdb_database,
                args.threads,
            )?;
            ctx.register_tables(&*benchmark, format)?;
            Ok(ctx)
        },
        |ctx, query_idx, format, query| {
            set_global_labels(vec![
                ("format", format.to_string()),
                ("benchmark_name", benchmark_name.clone()),
                ("query_idx", query_idx.to_string()),
            ]);

            // Make sure to reopen the duckdb connection between iterations
            if !args.reuse {
                ctx.reopen()?;
            }
            ctx.execute_query_result(query)
        },
    )?;

    if !args.explain {
        let benchmark_id = format!("duckdb-{}", benchmark.dataset_name());
        let writer = create_output_writer(&args.display_format, args.output_path, &benchmark_id)?;
        runner.export_to(&args.display_format, writer)?;
    }

    Ok(())
}
