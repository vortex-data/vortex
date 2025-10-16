// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data from the GitHub Archive.
//!
//! This dataset applies a bunch of events this way

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use log::info;
use parquet::arrow::async_writer::AsyncFileWriter;
use url::Url;
use vortex::compressor::CompactCompressor;
use vortex::file::{VortexWriteOptions, WriteStrategyBuilder};
use vortex_datafusion::VortexFormat;

use crate::benchmark_trait::Benchmark;
use crate::conversions::parquet_to_vortex;
use crate::engines::EngineCtx;
use crate::{BenchmarkDataset, Format, Target, idempotent, idempotent_async};

/// Template URL for raw JSON dataset
fn raw_json_url(hour: usize) -> String {
    assert!(hour <= 23);
    format!("https://data.gharchive.org/2024-10-01-{hour}.json.gz")
}

const QUERIES: &[&str] = &[
    "select * from events where payload.ref = 'refs/heads/main'",
    "select distinct repo.name from events where repo.name like 'spiraldb/%'",
    "select distinct org.id as org_id from events order by org_id limit 100",
    "select actor.login, count() as freq from events group by actor.login order by freq desc limit 10",
    "select actor.avatar_url from events where actor.login = 'renovate[bot]'",
];

pub struct GithubArchive {
    data_url: Url,
}

impl GithubArchive {
    pub fn new(data_url: Url) -> Self {
        Self { data_url }
    }
}

impl GithubArchive {
    fn json_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("json/")?
            .join("events.json.gz")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn parquet_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("parquet/")?
            .join("events.parquet")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn vortex_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("vortex-file-compressed/")?
            .join("events.vortex")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn vortex_compact_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("vortex-compact/")?
            .join("events.vortex")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

impl Benchmark for GithubArchive {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(QUERIES
            .iter()
            .map(|s| (s.to_string()))
            .enumerate()
            .collect())
    }

    fn generate_data(&self, target: &Target) -> anyhow::Result<()> {
        // Before downloading anything, make sure we are using a supported target.
        anyhow::ensure!(
            matches!(
                target.format,
                Format::Parquet | Format::OnDiskVortex | Format::VortexCompact
            ),
            "unsupported format for `fineweb` bench: {}",
            target.format
        );

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        // Write the JSON data to disk
        let json = rt.block_on(idempotent_async(
            &self.json_path()?,
            |json_path| async move {
                info!("Downloading GithubArchive JSON source files");
                // Download the files from gharchive.
                // They are all gzipped, so they can be concatenated into a single output file.
                let mut w = tokio::fs::File::create(json_path).await?;
                let client = reqwest::Client::new();
                for hour in 0..=23 {
                    let url = raw_json_url(hour);
                    info!("Downloading archive {url}");
                    let response = client
                        .get(url)
                        .send()
                        .await?
                        .error_for_status()
                        .map_err(|err| anyhow::anyhow!("error fetching gharchive data: {err}"))?;
                    let mut bytes = response.bytes_stream();

                    while let Some(next) = bytes.next().await {
                        let chunk = next?;
                        w.write(chunk).await?;
                    }
                }

                Ok(())
            },
        ))?;

        let json_path = json.display().to_string();

        let parquet = idempotent(&self.parquet_path()?, move |parquet_path| {
            let parquet = parquet_path.display().to_string();
            info!(
                "Converting GithubArchive JSON to Parquet with DuckDB @ {}",
                parquet_path.display()
            );
            let result = Command::new("duckdb")
                    .arg("-c")
                    .arg(format!("
                    CREATE TABLE events AS select * from read_ndjson_auto('{json_path}', ignore_errors = true);
                    COPY events TO '{parquet}' (FORMAT parquet);
                    "))
                    .spawn()?
                    .wait()?;

            if !result.success() {
                anyhow::bail!("DuckDB subprocess failed converting JSON to Parquet");
            }

            Ok(())
        })?;

        let target_path = match target.format {
            Format::Parquet => parquet,
            Format::OnDiskVortex => rt.block_on(idempotent_async(
                &self.vortex_path()?,
                |vortex_path| async move {
                    info!("Converting Parquet to Vortex with default compressor");
                    let array_stream = parquet_to_vortex(parquet)?;
                    let w = tokio::fs::File::create(vortex_path).await?;
                    VortexWriteOptions::default()
                        .write(w, array_stream)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to write to VortexWriter: {e}"))
                },
            ))?,
            Format::VortexCompact => rt.block_on(idempotent_async(
                &self.vortex_compact_path()?,
                |vortex_path| async move {
                    info!("Converting FineWeb to Vortex with Compact compressor");
                    let array_stream = parquet_to_vortex(parquet)?;
                    let w = tokio::fs::File::create(vortex_path).await?;
                    VortexWriteOptions::default()
                        .with_strategy(
                            WriteStrategyBuilder::new()
                                .with_compressor(CompactCompressor::default())
                                .build(),
                        )
                        .write(w, array_stream)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to write to VortexWriter: {e}"))
                },
            ))?,
            _ => anyhow::bail!("unsupported format for `gharchive` bench"),
        };

        info!("gharchive data generated in {}", target_path.display());

        Ok(())
    }

    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> anyhow::Result<()> {
        let dataset = self.dataset();

        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                dataset
                    .register_tables(&ctx.session, &self.data_url, format)
                    .await
            }
            EngineCtx::DuckDB(ctx) => ctx.register_tables(&self.data_url, format, &dataset),
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::GhArchive
    }

    fn dataset_name(&self) -> &str {
        "gharchive"
    }

    fn dataset_display(&self) -> String {
        "gharchive".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }
}

pub async fn register_table(
    session: &SessionContext,
    base_url: &Url,
    format: Format,
) -> anyhow::Result<()> {
    let table_path = base_url.join(&format!("{}/", format))?;
    info!("registering table for GHARCHIVE: {table_path}");
    let table_url = ListingTableUrl::try_new(table_path, None)?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(
            ListingOptions::new(match format {
                Format::Parquet => Arc::from(ParquetFormat::new()),
                Format::OnDiskVortex | Format::VortexCompact => Arc::from(VortexFormat::default()),
                _ => anyhow::bail!("unsupported format for `gharchive` bench: {}", format),
            })
            .with_session_config_options(session.state().config()),
        )
        .infer_schema(&session.state())
        .await?;
    let listing_table = Arc::new(ListingTable::try_new(config)?);
    session.register_table("events", listing_table)?;
    Ok(())
}
