// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_schema::Schema;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use glob::Pattern;
use tokio::fs::OpenOptions;
use tracing::info;
use url::Url;
use vortex::file::VortexWriteOptions;
use vortex_datafusion::VortexFormat;
use vortex_file::VortexLayoutStrategy;
use vortex_layout::layouts::compact::CompactCompressor;
use vortex_layout::scan::LocalExecutor;

use crate::conversions::parquet_to_vortex;
use crate::datasets::BenchmarkDataset;
use crate::idempotent_async;

pub async fn convert_parquet_to_vortex(
    input_path: &Path,
    dataset: &BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH { .. } => {
            // This is done on-demand by the register_vortex_file function
            Ok(())
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::convert_parquet_to_vortex(input_path).await
        }
        _ => todo!(),
    }
}

pub async fn convert_parquet_to_vortex_compact(
    input_path: &Path,
    dataset: &BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH { .. } => {
            // This is done on-demand by the register_vortex_compact_file function
            Ok(())
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::convert_parquet_to_vortex_compact(input_path).await
        }
        _ => todo!(),
    }
}

pub async fn register_parquet_files(
    session: &SessionContext,
    table_name: &str,
    file_url: &Url,
    glob: Option<Pattern>,
    schema: Option<Schema>,
    dataset: &BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH { .. } => {
            info!(
                "Registering table from {}, with glob {:?}",
                &file_url,
                glob.as_ref().map(|g| g.as_str()).unwrap_or("")
            );
            let format = Arc::new(ParquetFormat::new());

            let table_url = ListingTableUrl::try_new(file_url.clone(), glob)?;

            let config = ListingTableConfig::new(table_url).with_listing_options(
                ListingOptions::new(format).with_session_config_options(session.state().config()),
            );

            let config = if let Some(schema) = schema {
                config.with_schema(schema.into())
            } else {
                config.infer_schema(&session.state()).await?
            };

            let listing_table = Arc::new(ListingTable::try_new(config)?);

            session.register_table(table_name, listing_table)?;
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::register_parquet_files(
                session,
                table_name,
                file_url,
                &crate::clickbench::HITS_SCHEMA,
                glob,
            )?;
        }
        _ => todo!(),
    }

    Ok(())
}

pub async fn register_vortex_files(
    session: &SessionContext,
    table_name: &str,
    file_url: &Url,
    glob: Option<Pattern>,
    schema: Option<Schema>,
    dataset: &BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH { .. } | BenchmarkDataset::TpcDS { .. } => {
            info!(
                "Registering table from {}, with glob {:?}",
                &file_url,
                glob.as_ref().map(|g| g.as_str()).unwrap_or("")
            );
            let format = Arc::new(VortexFormat::default());
            let table_url = ListingTableUrl::try_new(file_url.clone(), glob)?;
            let config = ListingTableConfig::new(table_url).with_listing_options(
                ListingOptions::new(format).with_session_config_options(session.state().config()),
            );

            let config = if let Some(schema) = schema {
                config.with_schema(schema.into())
            } else {
                config.infer_schema(&session.state()).await?
            };

            let listing_table = Arc::new(ListingTable::try_new(config)?);

            session.register_table(table_name, listing_table)?;
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::register_vortex_files(
                session.clone(),
                table_name,
                file_url,
                schema,
                glob,
            )
            .await?;
        }
        BenchmarkDataset::PublicBi { .. } => todo!(),
    }

    Ok(())
}

pub async fn register_vortex_compact_files(
    session: &SessionContext,
    table_name: &str,
    file_url: &Url,
    glob: Option<Pattern>,
    schema: Option<Schema>,
    dataset: &BenchmarkDataset,
) -> Result<()> {
    match dataset {
        BenchmarkDataset::TpcH { .. } | BenchmarkDataset::TpcDS { .. } => {
            info!(
                "Registering vortex-compact table from {}, with glob {:?}",
                &file_url,
                glob.as_ref().map(|g| g.as_str()).unwrap_or("")
            );
            let format = Arc::new(VortexFormat::default());
            let table_url = ListingTableUrl::try_new(file_url.clone(), glob)?;
            let config = ListingTableConfig::new(table_url).with_listing_options(
                ListingOptions::new(format).with_session_config_options(session.state().config()),
            );

            let config = if let Some(schema) = schema {
                config.with_schema(schema.into())
            } else {
                config.infer_schema(&session.state()).await?
            };

            let listing_table = Arc::new(ListingTable::try_new(config)?);

            session.register_table(table_name, listing_table)?;
        }
        BenchmarkDataset::ClickBench { .. } => {
            crate::clickbench::register_vortex_compact_files(
                session.clone(),
                table_name,
                file_url,
                schema,
                glob,
            )
            .await?;
        }
        BenchmarkDataset::PublicBi { .. } => todo!(),
    }

    Ok(())
}

pub async fn parquet_file_to_vortex(parquet_path: &Path, vortex_path: &PathBuf) -> Result<()> {
    idempotent_async(vortex_path, async |vtx_file| {
        info!("Converting {:?} to Vortex format", parquet_path);

        let array_stream = parquet_to_vortex(parquet_path.to_path_buf())?;

        let f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&vtx_file)
            .await?;

        VortexWriteOptions::default().write(f, array_stream).await?;

        anyhow::Ok(())
    })
    .await?;

    Ok(())
}

pub async fn parquet_file_to_vortex_compact(parquet_path: &Path, vortex_compact_path: &PathBuf) -> Result<()> {
    idempotent_async(vortex_compact_path, async |vtx_file| {
        info!("Converting {:?} to Vortex compact format", parquet_path);

        let array_stream = parquet_to_vortex(parquet_path.to_path_buf())?;

        let f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&vtx_file)
            .await?;

        let executor = Arc::new(LocalExecutor);
        let compressor = CompactCompressor::default();
        let compact_strategy = VortexLayoutStrategy::compact_with_executor(executor, compressor);

        let compact_options = VortexWriteOptions::default()
            .with_strategy(compact_strategy);

        compact_options.write(f, array_stream).await?;

        anyhow::Ok(())
    })
    .await?;

    Ok(())
}
