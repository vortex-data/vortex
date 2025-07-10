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

use crate::conversions::parquet_to_vortex;
use crate::datasets::BenchmarkDataset;
use crate::{Format, idempotent_async};

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
            info!("Registering table from {}", &file_url);
            // ensure_parquet_file_exists(&object_store, file_url)?;
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
        BenchmarkDataset::ClickBench { single_file, .. } => {
            // For ClickBench, we use simplified pre-built Parquet registration
            let format = Arc::new(ParquetFormat::new());
            let mut parquet_path = dataset.format_path(Format::Parquet, file_url)?;

            if *single_file {
                parquet_path = parquet_path.join("hits.parquet")?;
            }

            info!("Registering table from {}", &parquet_path);
            let table_url = ListingTableUrl::parse(parquet_path)?;

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
            info!("Registering table from {}", &file_url);
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
        BenchmarkDataset::ClickBench { single_file, .. } => {
            crate::clickbench::register_vortex_files(
                session.clone(),
                table_name,
                file_url,
                schema,
                *single_file,
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
