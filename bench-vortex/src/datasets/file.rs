// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::Result;
use arrow_schema::Schema;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use glob::Pattern;
use tracing::info;
use url::Url;
use vortex_datafusion::VortexFormat;
#[cfg(feature = "lance")]
use {crate::Format, lance::datafusion::LanceTableProvider, lance::dataset::Dataset};

use crate::datasets::BenchmarkDataset;

pub async fn register_parquet_files(
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
        BenchmarkDataset::TpcH { .. }
        | BenchmarkDataset::TpcDS { .. }
        | BenchmarkDataset::Fineweb
        | BenchmarkDataset::GhArchive => {
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
        BenchmarkDataset::StatPopGen { .. } => todo!(),
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
        BenchmarkDataset::StatPopGen { .. } => todo!(),
        BenchmarkDataset::Fineweb => todo!(),
        BenchmarkDataset::GhArchive => todo!(),
    }

    Ok(())
}

#[cfg(feature = "lance")]
pub async fn register_lance_files(
    session: &SessionContext,
    table_name: &str,
    file_url: &Url,
    dataset: &BenchmarkDataset,
) -> Result<()> {
    let path = match dataset {
        BenchmarkDataset::TpcH { .. } | BenchmarkDataset::TpcDS { .. } => {
            file_url.join(&format!("{}.lance/", table_name))
        }
        BenchmarkDataset::ClickBench { .. } => {
            file_url.join(&format!("{}/hits.lance/", Format::Lance.name()))
        }
        _ => todo!("Lance support for {:?} not implemented", dataset),
    }?;

    let dataset = Dataset::open(path.as_str()).await?;
    let provider = LanceTableProvider::new(
        Arc::new(dataset),
        false, // with_row_id
        false, // with_row_addr
    );

    session.register_table(table_name, Arc::new(provider))?;
    info!("Successfully registered Lance table '{}'", table_name);

    Ok(())
}
