// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! PolarSignals benchmark implementation.

use anyhow::Result;
use url::Url;

use super::data::generate_polarsignals_parquet;
use super::schema::STACKTRACES_SCHEMA;
use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::FilePattern;
use crate::IdempotentPath;
use crate::QuerySource;
use crate::idempotent;

const FILE_NAME: &str = "stacktraces";

/// Benchmark modeled on PolarSignals profiling data to exercise scan-layer
/// performance (projection + filter pushdown) on deeply nested schemas.
pub struct PolarSignalsBenchmark {
    descriptor: BenchmarkDescriptor,
    /// Scale factor (1 = 1M rows).
    pub scale_factor: usize,
    /// Total number of rows in the dataset.
    pub n_rows: usize,
}

impl PolarSignalsBenchmark {
    /// Creates a new PolarSignalsBenchmark.
    ///
    /// `scale_factor` of 1 produces 1M rows.
    pub fn new(scale_factor: usize) -> Result<Self> {
        let n_rows = scale_factor * 1_000_000;
        let data_path = "polarsignals".to_data_path().join(format!("{n_rows}/"));
        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path"))?;

        let desc = BenchmarkDescriptor::new(
            "polarsignals",
            data_url,
            BenchmarkDataset::PolarSignals { n_rows },
        )
        .with_display(format!("polarsignals(sf={scale_factor})"))
        .with_queries(QuerySource::sql_file("polarsignals.sql"))
        .with_table("stacktraces", Some(STACKTRACES_SCHEMA.clone()))
        .with_file_pattern(FilePattern::Fixed(FILE_NAME));

        Ok(Self {
            descriptor: desc,
            scale_factor,
            n_rows,
        })
    }

    fn parquet_path(&self) -> Result<std::path::PathBuf> {
        self.descriptor
            .data_url
            .join("parquet/")?
            .join(&format!("{FILE_NAME}.parquet"))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("failed to convert data URL to filesystem path"))
    }
}

#[async_trait::async_trait]
impl Benchmark for PolarSignalsBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.descriptor.data_url.scheme() != "file" {
            anyhow::bail!(
                "unsupported URL scheme '{}' - only 'file://' URLs are supported",
                self.descriptor.data_url.scheme()
            );
        }

        let n_rows = self.n_rows;
        let parquet_path = self.parquet_path()?;
        idempotent(&parquet_path, |tmp_path| {
            tracing::info!(
                n_rows,
                path = %parquet_path.display(),
                "generating PolarSignals parquet data"
            );
            generate_polarsignals_parquet(n_rows, tmp_path)
        })?;
        Ok(())
    }
}
