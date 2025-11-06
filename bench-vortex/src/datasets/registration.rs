// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unified table registration for benchmarks
//!
//! This module provides a common interface for registering tables across
//! different file formats (Parquet, Vortex, Lance) without code duplication.

use std::sync::Arc;

use anyhow::Result;
use arrow_schema::Schema;
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use glob::Pattern;
use tracing::info;
use url::Url;
use vortex_datafusion::VortexFormat;

use crate::{Format, SESSION};

/// Creates a FileFormat instance for the given format type
pub fn create_file_format(format: Format) -> Result<Arc<dyn FileFormat>> {
    match format {
        Format::Parquet | Format::OnDiskDuckDB => Ok(Arc::new(ParquetFormat::new())),
        Format::OnDiskVortex | Format::VortexCompact => {
            Ok(Arc::new(VortexFormat::new(SESSION.clone())))
        }
        #[cfg(feature = "lance")]
        Format::Lance => {
            anyhow::bail!("Lance format uses LanceTableProvider, not FileFormat")
        }
        format => anyhow::bail!("Unsupported format for FileFormat creation: {}", format),
    }
}

/// Register a table using DataFusion's ListingTable abstraction
///
/// This is the unified registration method that works for Parquet, Vortex, and
/// Vortex Compact formats. It handles:
/// - URL construction with optional glob patterns
/// - Schema inference or explicit schema
/// - Session config options
///
/// # Arguments
/// * `session` - DataFusion session context
/// * `table_name` - Name to register the table as
/// * `file_url` - Base URL where files are located
/// * `glob` - Optional glob pattern for file matching (e.g., "*.parquet")
/// * `schema` - Optional explicit schema (will infer if not provided)
/// * `format` - FileFormat implementation (Parquet, Vortex, etc.)
pub async fn register_listing_table(
    session: &SessionContext,
    table_name: &str,
    file_url: &Url,
    glob: Option<Pattern>,
    schema: Option<Schema>,
    format: Arc<dyn FileFormat>,
) -> Result<()> {
    info!(
        "Registering table '{}' from {}, with glob {:?}",
        table_name,
        file_url,
        glob.as_ref().map(|g| g.as_str()).unwrap_or("")
    );

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

    Ok(())
}

#[cfg(feature = "lance")]
/// Register a Lance table using LanceTableProvider
///
/// Lance uses a different provider than FileFormat-based tables.
///
/// # Arguments
/// * `session` - DataFusion session context
/// * `table_name` - Name to register the table as
/// * `lance_path` - Full path to the Lance dataset (e.g., "file://path/table.lance/")
pub async fn register_lance_table(
    session: &SessionContext,
    table_name: &str,
    lance_path: &Url,
) -> Result<()> {
    use lance::datafusion::LanceTableProvider;
    use lance::dataset::Dataset;

    let dataset = Dataset::open(lance_path.as_str()).await?;
    let provider = LanceTableProvider::new(
        Arc::new(dataset),
        false, // with_row_id
        false, // with_row_addr
    );

    session.register_table(table_name, Arc::new(provider))?;
    info!(
        "Successfully registered Lance table '{}' from {}",
        table_name, lance_path
    );

    Ok(())
}
