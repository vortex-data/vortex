//! Benchmarking CLI for TPC-H.
//!
//! Queries execute one at a time using a sized pool of worker threads.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use anyhow::Context;
use bench_vortex::CTX;
use clap::{ArgAction, Parser, ValueEnum};
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use tokio::runtime::Builder;
use tracing::{debug, info, Level};
use tracing_subscriber::fmt::format::FmtSpan;
use url::Url;
use vortex_datafusion::persistent::format::VortexFormat;

#[derive(Parser, Debug)]
struct Cli {
    /// URL to root directory containing all TPC-H tables. One file per table.
    ///
    /// All TPC-H files must be generated before calling this CLI.
    source: Url,
    #[arg(long)]
    format: Format,

    /// Override the number of worker threads tokio uses for its thread pool.
    #[arg(long)]
    tokio_workers: Option<u8>,
    /// Allow specifying verbosity of logging.
    ///
    /// None: WARN
    /// -v:   INFO
    /// -vv:  DEBUG
    /// -vvv: TRACE
    #[arg(short, action = ArgAction::Count)]
    verbosity: u8,
}

#[derive(ValueEnum, Debug, Default, Clone, Copy)]
enum Format {
    #[default]
    Vortex,
    Parquet,
}

pub fn main() {
    let cli = Cli::parse();

    let (enabled, max_level) = match cli.verbosity {
        0 => (false, Level::WARN),
        1 => (true, Level::INFO),
        2 => (true, Level::DEBUG),
        _ => (true, Level::TRACE),
    };

    if enabled {
        tracing_subscriber::fmt()
            .with_max_level(max_level)
            .with_span_events(FmtSpan::CLOSE)
            .init();
        info!("logging enabled");
    }

    if let Ok(env_file) = dotenv::dotenv() {
        debug!("loaded environment from {}", env_file.display());
    }

    let mut runtime = Builder::new_multi_thread();
    if let Some(workers) = cli.tokio_workers {
        debug!("overriding tokio worker thread count to {workers}");

        runtime.worker_threads(workers as _);
    }

    runtime
        .enable_all()
        .build()
        .unwrap()
        .block_on(async_main(cli));
}

async fn async_main(cli: Cli) {
    // Create the datafusion context
    let df = SessionContext::default();

    let store = make_object_store(&cli.source);
    let mut base_url = Url::from(cli.source.clone());
    base_url.set_path("/");

    df.register_object_store(&base_url, store);

    // register all tables for TPC-H.
    for table in [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ] {
        register_table(table, cli.format, &cli.source, &df)
            .await
            .unwrap();
    }

    // For every requested query, execute it.
    let q1 = include_str!("../../tpch/q2.sql");
    df.sql(q1)
        .await
        .unwrap()
        .show()
        .await
        .unwrap();
}

fn make_object_store(source: &Url) -> Arc<dyn ObjectStore> {
    match source.scheme() {
        "s3" => {
            // Get bucket name.
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];
            Arc::new(
                AmazonS3Builder::from_env()
                    .with_bucket_name(bucket_name)
                    // .with_s3_express(true)
                    .build()
                    .unwrap(),
            )
        }
        "gcp" => {
            let bucket_name = &source[url::Position::BeforeHost..url::Position::AfterHost];

            Arc::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()
                    .unwrap(),
            )
        }
        _ => {
            // Just use local object store
            Arc::new(LocalFileSystem::default())
        }
    }
}

/// Register a new table at the given URL.
async fn register_table(
    name: &str,
    format: Format,
    base_url: &Url,
    df: &SessionContext,
) -> anyhow::Result<()> {
    let file_name = match format {
        Format::Vortex => format!("{name}.vortex"),
        Format::Parquet => format!("{name}.parquet"),
    };
    let file_url = base_url.join(file_name.as_str())?;

    debug!(table = name, url = file_url.as_str(), "registering table");

    let file_format = Arc::new(VortexFormat::new(&CTX));
    let table_url = ListingTableUrl::parse(file_url.as_str())?;
    info!(table_url = table_url.as_str(), "using table_url");

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(ListingOptions::new(file_format as _))
        .infer_schema(&df.state())
        .await
        .context("inferring schema")?;

    let listing_table = Arc::new(ListingTable::try_new(config)?);
    df.register_table(name, listing_table)?;

    Ok(())
}
