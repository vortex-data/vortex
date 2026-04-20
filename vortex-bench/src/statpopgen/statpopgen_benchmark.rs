// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::workspace_root;

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
    /// Base URL for the dataset location (must be a file:// URL for local datasets)
    pub data_url: Url,
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
    /// * `data_url` - Base URL pointing to the dataset location (must be a file:// URL)
    /// * `scale_factor` - Scale factor. The dataset will contain this many thousands of rows.
    ///
    /// # Returns
    /// A configured benchmark instance with pre-calculated expected row counts for query validation.
    pub fn new(scale_factor: u64) -> Result<Self> {
        let n_rows = scale_factor * 1000;
        let n_rows = usize::try_from(n_rows).map_err(|_| {
            anyhow::anyhow!(
                "Dataset size ({} rows) exceeds maximum supported size for this platform",
                n_rows
            )
        })?;

        let data_path = "statpopgen".to_data_path().join(format!("{n_rows}/"));

        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path?"))?;

        Ok(Self {
            data_url,
            scale_factor,
            n_rows: n_rows as u64,
        })
    }

    /// Returns the filesystem path to the Parquet dataset file.
    ///
    /// Constructs the path based on the configured data URL and number of rows.
    /// The path follows the pattern: `{data_url}/{n_rows}/parquet/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.parquet`
    pub fn parquet_path(&self) -> Result<PathBuf> {
        self.data_url
            .join("parquet/")?
            .join(&format!("{}.parquet", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    /// Returns the filesystem path to the compressed Vortex dataset file.
    ///
    /// Constructs the path based on the configured data URL and number of rows.
    /// The path follows the pattern: `{data_url}/{n_rows}/vortex-file-compressed/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vortex`
    pub fn vortex_path(&self) -> Result<PathBuf> {
        self.data_url
            .join("vortex-file-compressed/")?
            .join(&format!("{}.vortex", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    /// Returns the filesystem path to the compacted Vortex dataset file.
    ///
    /// Constructs the path based on the configured data URL and number of rows.
    /// The path follows the pattern: `{data_url}/{n_rows}/vortex-compact/{StatPopGenBenchmark::FILE_NAME}.vortex`
    pub fn vortex_compact_path(&self) -> Result<PathBuf> {
        self.data_url
            .join("vortex-compact/")?
            .join(&format!("{}.vortex", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for StatPopGenBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("statpopgen")
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .trim()
            .split_terminator(";")
            .map(str::to_string)
            .enumerate()
            .collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            anyhow::bail!(
                "Unsupported URL scheme '{}' - only 'file://' URLs are supported for local datasets",
                self.data_url.scheme()
            );
        }

        self.download_parquet().await?;
        Ok(())
    }

    #[expect(clippy::cast_possible_truncation)]
    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        let n_rows = self.n_rows as usize;
        match self.scale_factor {
            1 => Some(vec![
                1, 1, n_rows, n_rows, n_rows, n_rows, n_rows, 1, 47, 891,
            ]),
            10 => Some(vec![
                1, 1, n_rows, n_rows, n_rows, n_rows, n_rows, 1, 47, 8507,
            ]),
            100 => Some(vec![
                1, 1, n_rows, n_rows, n_rows, n_rows, n_rows, 1, 47, 85877,
            ]),
            _ => None,
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::StatPopGen {
            n_rows: self.n_rows,
        }
    }

    fn dataset_name(&self) -> &str {
        "statpopgen"
    }

    fn dataset_display(&self) -> String {
        "statpopgen".to_string()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("statpopgen", None)]
    }

    #[expect(clippy::expect_used, clippy::unwrap_in_result)]
    fn pattern(&self, _table_name: &str, format: Format) -> Option<glob::Pattern> {
        Some(
            format!("{}.{}", Self::FILE_NAME, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
