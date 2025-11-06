// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unified table registration using dataset metadata
//!
//! This module provides helpers to register tables using the
//! DatasetMetadata trait, reducing duplication across benchmarks.

use anyhow::Result;
use datafusion::prelude::SessionContext;
use glob::Pattern;
use url::Url;

use crate::Format;
use super::metadata::DatasetMetadata;
use super::registration::{create_file_format, register_listing_table};

/// Register all tables for a dataset using metadata
///
/// This function uses the DatasetMetadata trait to determine
/// what tables to register and their file patterns.
pub async fn register_dataset_tables(
    session: &SessionContext,
    dataset: &dyn DatasetMetadata,
    base_url: &Url,
    format: Format,
) -> Result<()> {
    // Special case for Arrow format (in-memory)
    let file_format = if format == Format::Arrow {
        Format::Parquet  // Arrow loads Parquet files into memory
    } else {
        format
    };

    let format_url = base_url.join(&format!("{}/", file_format.name()))?;
    let file_format_instance = create_file_format(file_format)?;

    for table_info in dataset.tables() {
        // Create glob pattern from table info
        let pattern = if table_info.file_pattern.contains('*') {
            table_info.file_pattern.clone()
        } else {
            // If no wildcard, assume it's a prefix
            format!("{}*.{}", table_info.file_pattern, file_format.ext())
        };

        let glob = Some(Pattern::new(&pattern)?);

        match format {
            Format::Arrow => {
                // For Arrow format, we need special in-memory handling
                // This would need to be implemented based on the existing register_arrow function
                todo!("Arrow format registration needs special handling")
            }
            #[cfg(feature = "lance")]
            Format::Lance => {
                // Lance has its own registration path
                use super::registration;
                let lance_path = format_url.join(&format!("{}.lance/", table_info.name))?;
                registration::register_lance_table(session, &table_info.name, &lance_path).await?;
            }
            _ => {
                // Standard file format registration
                register_listing_table(
                    session,
                    &table_info.name,
                    &format_url,
                    glob,
                    table_info.schema_hint.clone(),
                    file_format_instance.clone(),
                )
                .await?;
            }
        }
    }

    Ok(())
}

/// Helper to create a dataset-aware URL
///
/// This combines the base URL with the dataset's name and variant
/// to create the appropriate path.
pub fn dataset_url(base_url: &Url, dataset: &dyn DatasetMetadata) -> Result<Url> {
    let path = if dataset.variant().is_empty() {
        dataset.name().to_string()
    } else {
        format!("{}/{}", dataset.name(), dataset.variant())
    };

    Ok(base_url.join(&format!("{}/", path))?)
}