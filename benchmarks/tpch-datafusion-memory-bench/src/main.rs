// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(feature = "dhat"))]
compile_error!("tpch-datafusion-memory-bench requires the `dhat` feature");

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::prelude::SessionContext;
use indicatif::ProgressBar;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkOutput;
use vortex_bench::CompactionStrategy;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Target;
use vortex_bench::conversions::convert_parquet_directory_to_vortex;
use vortex_bench::dhat::start_heap_profiling;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::display::render_table;
use vortex_bench::measurements::NamedMeasurement;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::tpch::benchmark::TpcHBenchmark;
use vortex_bench::utils::constants::STORAGE_NVME;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "1.0")]
    scale_factor: String,
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![Format::Parquet, Format::OnDiskVortex]
    )]
    formats: Vec<Format>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(short, long)]
    output_path: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, args.tracing)?;

    run_tpch_memory(
        args.scale_factor,
        args.formats,
        args.display_format,
        args.output_path,
    )
    .await
}

async fn run_tpch_memory(
    scale_factor: String,
    formats: Vec<Format>,
    display_format: DisplayFormat,
    output_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    for format in &formats {
        anyhow::ensure!(
            matches!(format, Format::Parquet | Format::OnDiskVortex),
            "TPC-H memory benchmark only supports parquet and vortex formats",
        );
    }

    let benchmark = TpcHBenchmark::new(scale_factor, None)?;
    benchmark.generate_base_data().await?;

    let queries = benchmark.queries()?;
    let expected_row_counts = benchmark.expected_row_counts().map(|rows| rows.to_vec());
    let progress = ProgressBar::new((queries.len() * formats.len()) as u64);
    let targets = formats
        .iter()
        .map(|format| Target::new(Engine::DataFusion, *format))
        .collect::<Vec<_>>();
    let mut measurements = Vec::with_capacity(formats.len());

    if formats.contains(&Format::OnDiskVortex) {
        let base_path = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;
        convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Default).await?;
    }

    for format in formats {
        let session = datafusion_bench::get_session_context();
        datafusion_bench::make_object_store(&session, benchmark.data_url())?;
        register_tables(&session, &benchmark, format).await?;

        let profiler = start_heap_profiling()?;
        for (query_idx, query) in &queries {
            tracing::info!(query_idx, %format, "Running TPC-H query");
            let batches = session.sql(query).await?.collect().await?;
            let row_count = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();

            if let Some(expected) = &expected_row_counts {
                assert_eq!(
                    row_count, expected[*query_idx],
                    "Row count mismatch for query {query_idx}",
                );
            }

            progress.inc(1);
        }

        let stats = profiler.finish();
        measurements.push(NamedMeasurement {
            name: format!("tpch peak memory/datafusion:{}", format.name()),
            target: Target::new(Engine::DataFusion, format),
            unit: "MiB".into(),
            value: stats.max_mib(),
            storage: Some(STORAGE_NVME.to_string()),
        });
    }
    progress.finish();
    let output = BenchmarkOutput::with_path("tpch-datafusion-memory", output_path);
    let mut writer = output.create_writer()?;

    match display_format {
        DisplayFormat::Table => render_table(&mut writer, measurements, &targets)?,
        DisplayFormat::GhJson => print_measurements_json(&mut writer, measurements)?,
    }

    Ok(())
}

async fn register_tables(
    session: &SessionContext,
    benchmark: &impl Benchmark,
    format: Format,
) -> anyhow::Result<()> {
    let benchmark_base = benchmark.data_url().join(&format!("{}/", format.name()))?;
    let file_format = datafusion_bench::format_to_df_format(format);

    for table in benchmark.table_specs() {
        let table_url = ListingTableUrl::try_new(
            benchmark_base.clone(),
            benchmark.pattern(table.name, format),
        )?;

        let listing_options = ListingOptions::new(file_format.clone())
            .with_session_config_options(session.state().config());
        let mut config = ListingTableConfig::new(table_url).with_listing_options(listing_options);

        config = match table.schema.as_ref() {
            Some(schema) => config.with_schema(Arc::new(schema.clone())),
            None => config.infer_schema(&session.state()).await?,
        };

        let listing_table = Arc::new(ListingTable::try_new(config)?);
        session.register_table(table.name, listing_table)?;
    }

    Ok(())
}
