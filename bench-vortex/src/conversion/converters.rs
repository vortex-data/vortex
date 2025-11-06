// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete implementations of format converters

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::fs;
use tokio::fs::OpenOptions;
use tracing::info;
use vortex::file::WriteOptionsSessionExt;

use crate::conversion::{
    CompactionStrategy as ConversionCompactionStrategy, ConversionOptions, FormatConverter,
};
use crate::conversions::parquet_to_vortex;
use crate::{CompactionStrategy, Format, SESSION};

/// Converter from Parquet to Vortex format
#[derive(Clone)]
pub struct ParquetToVortexConverter {
    source_format: Format,
    target_format: Format,
}

impl ParquetToVortexConverter {
    pub fn new(vortex_format: Format) -> Self {
        assert!(
            vortex_format == Format::OnDiskVortex || vortex_format == Format::VortexCompact,
            "Target format must be a Vortex format"
        );
        Self {
            source_format: Format::Parquet,
            target_format: vortex_format,
        }
    }

    async fn convert_single_file(
        &self,
        source_file: &Path,
        target_dir: &Path,
        compaction: ConversionCompactionStrategy,
    ) -> Result<()> {
        // Generate output filename
        let filename = source_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid source file name"))?;

        let output_file = target_dir.join(format!("{}.{}", filename, self.target_format.ext()));

        // Skip if already exists
        if output_file.exists() && !output_file.is_dir() {
            info!("Skipping {}, already exists", output_file.display());
            return Ok(());
        }

        info!(
            "Converting {} to {}",
            source_file.display(),
            output_file.display()
        );

        // Read Parquet file into Vortex arrays

        let array_stream = parquet_to_vortex(source_file.to_path_buf())?;

        // Open output file
        let mut output = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&output_file)
            .await?;

        // Apply compaction strategy

        let strategy = match (self.target_format, compaction) {
            (Format::VortexCompact, _) | (_, ConversionCompactionStrategy::Compact) => {
                CompactionStrategy::Compact
            }
            _ => CompactionStrategy::Default,
        };

        let write_options = strategy.apply_options(SESSION.write_options());

        // Write to Vortex format
        write_options.write(&mut output, array_stream).await?;

        Ok(())
    }
}

#[async_trait]
impl FormatConverter for ParquetToVortexConverter {
    async fn convert(
        &self,
        source_path: &Path,
        target_path: &Path,
        options: &ConversionOptions,
    ) -> Result<()> {
        // Ensure target directory exists
        fs::create_dir_all(target_path).await?;

        // Find all Parquet files in source directory
        let pattern = source_path.join("*.parquet");
        let pattern_str = pattern
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;

        let files: Vec<_> = glob::glob(pattern_str)?.filter_map(Result::ok).collect();

        if files.is_empty() {
            anyhow::bail!("No Parquet files found in {}", source_path.display());
        }

        info!(
            "Converting {} Parquet files to {}",
            files.len(),
            self.target_format
        );

        // Convert files with optional parallelism
        let parallelism = options.parallelism.unwrap_or(1);

        if parallelism > 1 {
            // Parallel conversion using tokio tasks
            let semaphore = Arc::new(tokio::sync::Semaphore::new(parallelism));
            let mut handles = Vec::new();

            for file in files {
                let permit = semaphore.clone().acquire_owned().await?;
                let target_dir = target_path.to_path_buf();
                let compaction = options.compaction;
                let converter = self.clone();

                let handle = tokio::spawn(async move {
                    let _permit = permit; // Hold permit until done
                    converter
                        .convert_single_file(&file, &target_dir, compaction)
                        .await
                });
                handles.push(handle);
            }

            // Wait for all conversions to complete
            for handle in handles {
                handle.await??;
            }
        } else {
            // Sequential conversion
            for file in files {
                self.convert_single_file(&file, target_path, options.compaction)
                    .await?;
            }
        }

        Ok(())
    }

    fn supports(&self, source_format: Format, target_format: Format) -> bool {
        source_format == self.source_format && target_format == self.target_format
    }

    fn name(&self) -> &str {
        match self.target_format {
            Format::OnDiskVortex => "Parquet to Vortex",
            Format::VortexCompact => "Parquet to Vortex Compact",
            _ => unreachable!(),
        }
    }

    fn source_format(&self) -> Format {
        self.source_format
    }

    fn target_format(&self) -> Format {
        self.target_format
    }
}

#[cfg(feature = "lance")]
/// Converter from Parquet to Lance format
pub struct ParquetToLanceConverter;

#[cfg(feature = "lance")]
impl ParquetToLanceConverter {
    pub fn new() -> Self {
        Self
    }

    async fn convert_table(
        &self,
        table_name: &str,
        source_path: &Path,
        target_path: &Path,
    ) -> Result<()> {
        use datafusion::prelude::SessionContext;
        use lance::dataset::{WriteMode, WriteOptions, WriteStrategyBuilder};
        use url::Url;

        use crate::datasets::registration::{create_file_format, register_listing_table};

        let lance_path = target_path.join(format!("{}.lance", table_name));

        // Skip conversion if target already exists
        if lance_path.exists() {
            info!("Skipping {}, already exists", lance_path.display());
            return Ok(());
        }

        info!("Converting {} to Lance format", table_name);

        // Create a temporary DataFusion context for conversion
        let ctx = SessionContext::new();

        // Register the Parquet files as a table
        let file_url =
            Url::from_directory_path(source_path).map_err(|_| anyhow::anyhow!("Invalid path"))?;

        // Use the registration helper to register Parquet files

        let format = create_file_format(Format::Parquet)?;

        register_listing_table(
            &ctx,
            table_name,
            &file_url,
            Some(glob::Pattern::new(&format!("{}_*.parquet", table_name))?),
            None,
            format,
        )
        .await?;

        // Read the table and convert Utf8View to Utf8 if needed
        let df = ctx.table(table_name).await?;
        let batch_stream = df.execute_stream().await?;

        // Write to Lance format
        let write_options = WriteOptions {
            mode: WriteMode::Create,
            write_strategy: WriteStrategyBuilder::default()
                .file_size(512 * 1024 * 1024) // 512MB files
                .build(),
            ..Default::default()
        };

        lance::dataset::write(batch_stream, lance_path.to_str().unwrap(), write_options).await?;

        Ok(())
    }
}

#[cfg(feature = "lance")]
#[async_trait]
impl FormatConverter for ParquetToLanceConverter {
    async fn convert(
        &self,
        source_path: &Path,
        target_path: &Path,
        _options: &ConversionOptions,
    ) -> Result<()> {
        // Lance conversion is typically done per-table
        // We need to infer table names from the file patterns

        use std::collections::HashSet;

        fs::create_dir_all(target_path).await?;

        // Find all unique table names from Parquet files
        let pattern = source_path.join("*.parquet");
        let pattern_str = pattern
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;

        let mut table_names = HashSet::new();
        for entry in glob::glob(pattern_str)? {
            if let Ok(path) = entry {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    // Extract table name (everything before the last underscore)
                    if let Some(pos) = stem.rfind('_') {
                        table_names.insert(stem[..pos].to_string());
                    } else {
                        table_names.insert(stem.to_string());
                    }
                }
            }
        }

        if table_names.is_empty() {
            anyhow::bail!("No Parquet files found in {}", source_path.display());
        }

        info!("Converting {} tables to Lance format", table_names.len());

        for table_name in table_names {
            self.convert_table(&table_name, source_path, target_path)
                .await?;
        }

        Ok(())
    }

    fn supports(&self, source_format: Format, target_format: Format) -> bool {
        source_format == Format::Parquet && target_format == Format::Lance
    }

    fn name(&self) -> &str {
        "Parquet to Lance"
    }

    fn source_format(&self) -> Format {
        Format::Parquet
    }

    fn target_format(&self) -> Format {
        Format::Lance
    }
}

/// Identity converter (no conversion needed)
pub struct IdentityConverter {
    format: Format,
}

impl IdentityConverter {
    pub fn new(format: Format) -> Self {
        Self { format }
    }
}

#[async_trait]
impl FormatConverter for IdentityConverter {
    async fn convert(
        &self,
        _source_path: &Path,
        _target_path: &Path,
        _options: &ConversionOptions,
    ) -> Result<()> {
        // No conversion needed - source and target are the same format
        Ok(())
    }

    fn supports(&self, source_format: Format, target_format: Format) -> bool {
        source_format == self.format && target_format == self.format
    }

    fn name(&self) -> &str {
        "Identity (no conversion)"
    }

    fn source_format(&self) -> Format {
        self.format
    }

    fn target_format(&self) -> Format {
        self.format
    }
}
