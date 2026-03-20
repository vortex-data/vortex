// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::FilePattern;
use crate::IdempotentPath;
use crate::QuerySource;

/// Statistical population genetics benchmark implementation.
///
/// This benchmark runs genomic analysis queries against variant data from the
/// gnomAD (Genome Aggregation Database) dataset. It supports multiple data formats
/// including Parquet and Vortex for performance comparison.
///
/// The benchmark consists of queries that perform typical population genetics analyses
/// such as allele frequency calculations, Hardy-Weinberg equilibrium tests, and
/// variant filtering operations.
pub struct StatPopGenBenchmark {
    descriptor: BenchmarkDescriptor,
    /// The scale factor. The dataset contains this many thousands of rows.
    pub scale_factor: u64,
    /// The number of rows in the dataset.
    pub n_rows: u64,
}

impl StatPopGenBenchmark {
    pub const FILE_NAME: &str = "gnomad.genomes.v3.1.2.hgdp_tgp.chr21";

    /// Creates a new StatPopGenBenchmark instance.
    ///
    /// # Arguments
    /// * `scale_factor` - Scale factor. The dataset will contain this many thousands of rows.
    pub fn new(scale_factor: u64) -> Result<Self> {
        let n_rows = scale_factor * 1000;
        let n_rows_usize = usize::try_from(n_rows).map_err(|_| {
            anyhow::anyhow!(
                "Dataset size ({} rows) exceeds maximum supported size for this platform",
                n_rows
            )
        })?;

        let data_path = "statspopgen"
            .to_data_path()
            .join(format!("{n_rows_usize}/"));

        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path?"))?;

        #[allow(clippy::cast_possible_truncation)]
        let expected_row_counts = match scale_factor {
            1 => Some(vec![
                1,
                1,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                1,
                47,
                891,
            ]),
            10 => Some(vec![
                1,
                1,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                1,
                47,
                8507,
            ]),
            100 => Some(vec![
                1,
                1,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                n_rows_usize,
                1,
                47,
                85877,
            ]),
            _ => None,
        };

        let mut desc = BenchmarkDescriptor::new(
            "statpopgen",
            data_url,
            BenchmarkDataset::StatPopGen { n_rows },
        )
        .with_display("statpopgen".to_string())
        .with_queries(QuerySource::sql_file("statpopgen.sql"))
        .with_table("statpopgen", None)
        .with_file_pattern(FilePattern::Fixed(Self::FILE_NAME));

        if let Some(counts) = expected_row_counts {
            desc = desc.with_expected_row_counts(counts);
        }

        Ok(Self {
            descriptor: desc,
            scale_factor,
            n_rows,
        })
    }

    /// Returns the filesystem path to the Parquet dataset file.
    pub fn parquet_path(&self) -> Result<PathBuf> {
        self.descriptor
            .data_url
            .join("parquet/")?
            .join(&format!("{}.parquet", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    /// Returns the filesystem path to the compressed Vortex dataset file.
    pub fn vortex_path(&self) -> Result<PathBuf> {
        self.descriptor
            .data_url
            .join("vortex-file-compressed/")?
            .join(&format!("{}.vortex", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    /// Returns the filesystem path to the compacted Vortex dataset file.
    pub fn vortex_compact_path(&self) -> Result<PathBuf> {
        self.descriptor
            .data_url
            .join("vortex-compact/")?
            .join(&format!("{}.vortex", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for StatPopGenBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.descriptor.data_url.scheme() != "file" {
            anyhow::bail!(
                "Unsupported URL scheme '{}' - only 'file://' URLs are supported for local datasets",
                self.descriptor.data_url.scheme()
            );
        }

        self.download_parquet().await?;
        Ok(())
    }
}
