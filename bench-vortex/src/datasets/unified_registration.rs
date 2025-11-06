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

use super::metadata::DatasetMetadata;
use super::registration::{create_file_format, register_listing_table};
use crate::Format;

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
        Format::Parquet // Arrow loads Parquet files into memory
    } else {
        format
    };

    let format_url = base_url.join(&format!("{}/", file_format.name()))?;
    let file_format_instance = create_file_format(file_format)?;

    for table_info in dataset.tables() {
        // Create glob pattern from table info
        let pattern = if table_info.file_pattern.contains('*') {
            // Pattern already has wildcard - append extension if needed
            if table_info.file_pattern.contains('.') {
                // Already has an extension
                table_info.file_pattern.clone()
            } else {
                // Add the format extension
                format!("{}.{}", table_info.file_pattern, file_format.ext())
            }
        } else {
            // No wildcard, assume it's a prefix
            format!("{}*.{}", table_info.file_pattern, file_format.ext())
        };

        let glob = Some(Pattern::new(&pattern)?);

        match format {
            Format::Arrow => {
                // For Arrow format, load Parquet files into memory as RecordBatches
                // This gives us pure in-memory query performance

                // Arrow format requires local filesystem (no glob support in object store)
                if format_url.scheme() != "file" {
                    anyhow::bail!("Arrow format only supports local filesystem");
                }

                // Read parquet files and convert to in-memory table
                // Arrow always reads from Parquet files
                let arrow_pattern = if table_info.file_pattern.contains('*') {
                    if table_info.file_pattern.contains('.') {
                        table_info.file_pattern.clone()
                    } else {
                        format!("{}.{}", table_info.file_pattern, Format::Parquet.ext())
                    }
                } else {
                    format!("{}*.{}", table_info.file_pattern, Format::Parquet.ext())
                };
                let glob_pattern = arrow_pattern.as_str();

                let path = format_url.path().to_string() + glob_pattern;
                let df = session.read_parquet(&path, Default::default()).await?;

                // Get schema before consuming df
                let schema = df.schema().inner().clone();

                // Collect all batches into memory
                let batches = df.collect().await?;

                // Create a memory table from the batches
                use std::sync::Arc;

                use datafusion::datasource::MemTable;

                let provider = MemTable::try_new(schema, vec![batches])?;

                session.register_table(&table_info.name, Arc::new(provider))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::configs::TpcHDataset;

    /// Helper function that mimics the pattern building logic
    fn build_pattern(file_pattern: &str, format: Format) -> String {
        let file_format = if format == Format::Arrow {
            Format::Parquet // Arrow loads Parquet files
        } else {
            format
        };

        if file_pattern.contains('*') {
            // Pattern already has wildcard - append extension if needed
            if file_pattern.contains('.') {
                // Already has an extension
                file_pattern.to_string()
            } else {
                // Add the format extension
                format!("{}.{}", file_pattern, file_format.ext())
            }
        } else {
            // No wildcard, assume it's a prefix
            format!("{}*.{}", file_pattern, file_format.ext())
        }
    }

    #[test]
    fn test_file_pattern_building() {
        // Test various pattern formats
        assert_eq!(
            build_pattern("customer_*", Format::Parquet),
            "customer_*.parquet"
        );
        assert_eq!(
            build_pattern("customer_*", Format::OnDiskVortex),
            "customer_*.vortex"
        );
        assert_eq!(build_pattern("*", Format::Parquet), "*.parquet");
        assert_eq!(build_pattern("*", Format::OnDiskVortex), "*.vortex");

        // Test that existing extensions are preserved
        assert_eq!(
            build_pattern("*.parquet", Format::OnDiskVortex),
            "*.parquet"
        );

        // Test patterns without wildcards
        assert_eq!(
            build_pattern("customer", Format::Parquet),
            "customer*.parquet"
        );
        assert_eq!(
            build_pattern("customer", Format::OnDiskVortex),
            "customer*.vortex"
        );

        // Test Arrow format (should use Parquet extension)
        assert_eq!(
            build_pattern("customer_*", Format::Arrow),
            "customer_*.parquet"
        );
        assert_eq!(build_pattern("*", Format::Arrow), "*.parquet");
    }

    #[test]
    fn test_tpch_patterns() {
        let dataset = TpcHDataset {
            scale_factor: "1.0".to_string(),
        };

        let tables = dataset.tables();
        assert!(!tables.is_empty());

        // Check that TPC-H patterns work with different formats
        for table_info in tables {
            // Should generate correct patterns for different formats
            let parquet_pattern = build_pattern(&table_info.file_pattern, Format::Parquet);
            let vortex_pattern = build_pattern(&table_info.file_pattern, Format::OnDiskVortex);

            // Patterns should have the right extensions
            assert!(parquet_pattern.ends_with(".parquet"));
            assert!(vortex_pattern.ends_with(".vortex"));

            // Patterns should contain the table name
            assert!(parquet_pattern.contains(&table_info.name));
            assert!(vortex_pattern.contains(&table_info.name));
        }
    }

    #[test]
    fn test_wildcard_patterns() {
        // Test FineWeb/GhArchive style patterns
        let patterns = vec!["*", "*.parquet", "events_*", "fineweb_*"];

        for pattern in patterns {
            let parquet_result = build_pattern(pattern, Format::Parquet);
            let vortex_result = build_pattern(pattern, Format::OnDiskVortex);

            // If pattern already has extension, it should be preserved
            if pattern.contains(".parquet") {
                assert_eq!(parquet_result, pattern);
                assert_eq!(vortex_result, pattern); // Preserved even for different format
            } else if pattern == "*" {
                assert_eq!(parquet_result, "*.parquet");
                assert_eq!(vortex_result, "*.vortex");
            } else if pattern.contains('*') {
                assert!(parquet_result.ends_with(".parquet"));
                assert!(vortex_result.ends_with(".vortex"));
            }
        }
    }
}
