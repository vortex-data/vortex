// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
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
use crate::{BenchmarkDataset, Format, Target, idempotent_async};

/// URL to the sample file
const SAMPLE_URL: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/v1.4.0/sample/10BT/001_00000.parquet";

/// Some basic string-focused queries.
const QUERIES: &[&str] = &[
    // simple summary
    "select count(distinct dump) from fineweb",
    // selective string equality filter
    "select * from fineweb where dump = 'CC-MAIN-2016-30'",
    // LIKE with prefix filter
    "select * from fineweb where date like '2020-10-%'",
    // LIKE with simple containment filter
    "select * from fineweb where url like '%google%' and text like '%Google%'",
    // LIKE with larger containment filter
    "select * from fineweb where url like '%.google.%' or text like '% Google %'",
    "select * from fineweb where text like '% vortex %'",
    // More LIKE filters
    "select * from fineweb where url like '%espn%' and language = 'en' and language_score > 0.92",
    "select * from fineweb where url like '%espn%' or url like '%www.espn.go.com%' or url like '%espn.go.com%'",
    // no results, stats cannot prune but tokenized bloom filters could
    "select * from fineweb where file_path like '%/CC-MAIN-2014-%'",
];

/// A benchmark using the HuggingFace FineWeb dataset.
///
/// This is a very string-heavy dataset, and exercises dictionary and FSST encoding heavily.
///
/// The queries for this benchmark are hand-crafted to showcase just how many of these we have here.
pub struct Fineweb {
    data_url: Url,
}

impl Fineweb {
    pub fn new(data_url: Url) -> Self {
        Self { data_url }
    }
}

impl Fineweb {
    fn parquet_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("parquet/")?
            .join("sample.parquet")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn vortex_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("vortex-file-compressed/")?
            .join("sample.vortex")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn vortex_compact_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("vortex-compact/")?
            .join("sample.vortex")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

impl Benchmark for Fineweb {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(QUERIES.iter().map(|s| s.to_string()).enumerate().collect())
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

        // Write the parquet data to disk.
        let parquet = rt.block_on(idempotent_async(
            &self.parquet_path()?,
            |parquet_path| async move {
                info!("Downloading FineWeb Parquet source from HuggingFace");
                // Download the file from HuggingFace snapshot
                let client = reqwest::Client::new();
                let response = client
                    .get(SAMPLE_URL)
                    .send()
                    .await?
                    .error_for_status()
                    .map_err(|err| {
                        anyhow::anyhow!("error fetching fineweb sample from HuggingFace: {err}")
                    })?;

                // On success, stream the response body to file.
                let mut bytes = response.bytes_stream();
                let mut w = tokio::fs::File::create(parquet_path).await?;

                while let Some(next) = bytes.next().await {
                    let chunk = next?;
                    w.write(chunk).await?;
                }

                Ok(())
            },
        ))?;

        let target_dir = match target.format {
            Format::Parquet => {
                // Nothing to do here
                parquet
            }
            Format::OnDiskVortex => rt.block_on(idempotent_async(
                &self.vortex_path()?,
                |vortex_path| async move {
                    info!("Converting FineWeb to Vortex with default compressor");
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
            _ => {
                anyhow::bail!("unsupported format for `fineweb` bench: {}", target.format)
            }
        };

        info!("fineweb data generated in {}", target_dir.display());

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
        BenchmarkDataset::Fineweb
    }

    fn dataset_name(&self) -> &str {
        "fineweb"
    }

    fn dataset_display(&self) -> String {
        "fineweb".to_owned()
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
    info!("registering table for FINEWEB: {table_path}");
    let table_url = ListingTableUrl::try_new(table_path, None)?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(
            ListingOptions::new(match format {
                Format::Parquet => Arc::from(ParquetFormat::new()),
                Format::OnDiskVortex | Format::VortexCompact => Arc::from(VortexFormat::default()),
                _ => anyhow::bail!("unsupported format for `fineweb` bench: {}", format),
            })
            .with_session_config_options(session.state().config()),
        )
        .infer_schema(&session.state())
        .await?;
    let listing_table = Arc::new(ListingTable::try_new(config)?);
    session.register_table("fineweb", listing_table)?;
    Ok(())
}
