// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use tracing::info;
use url::Url;
use vortex_datafusion::VortexFormat;

use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::{BenchmarkDataset, Format, SESSION, Target};

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
    /// Expected row counts for each benchmark query, used for result validation
    pub expected_row_counts: Option<Vec<usize>>,
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
    pub fn new(data_url: Url, scale_factor: u64) -> Result<Self> {
        let n_rows = scale_factor * 1000;
        let n_rows = usize::try_from(n_rows).map_err(|_| {
            anyhow::anyhow!(
                "Dataset size ({} rows) exceeds maximum supported size for this platform",
                n_rows
            )
        })?;

        // The number of rows returned by the filter (the last query) varies by number of rows.
        let expected_row_counts = match scale_factor {
            1 => Some(vec![
                1, 1, n_rows, n_rows, n_rows, n_rows, n_rows, 1, 47, 891,
            ]),
            10 => Some(vec![
                1, 1, n_rows, n_rows, n_rows, n_rows, n_rows, 1, 47, 8507,
            ]),
            100 => Some(vec![
                1, 1, n_rows, n_rows, n_rows, n_rows, n_rows, 1, 47, 85877,
            ]),
            _ => {
                info!(
                    "Expected row counts are not available for scale factor {}",
                    scale_factor
                );
                None
            }
        };

        Ok(Self {
            data_url,
            scale_factor,
            n_rows: n_rows as u64,
            expected_row_counts,
        })
    }

    /// Returns the filesystem path to the Parquet dataset file.
    ///
    /// Constructs the path based on the configured data URL and number of rows.
    /// The path follows the pattern: `{data_url}/{n_rows}/parquet/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.parquet`
    pub fn parquet_path(&self) -> Result<PathBuf> {
        self.data_url
            .join(&(self.n_rows.to_string() + "/parquet/"))?
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
            .join(&(self.n_rows.to_string() + "/vortex-file-compressed/"))?
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
            .join(&(self.n_rows.to_string() + "/vortex-compact/"))?
            .join(&format!("{}.vortex", StatPopGenBenchmark::FILE_NAME))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

/// Registers a table with DataFusion for the specified format.
///
/// Creates and configures a ListingTable that points to the dataset file in the
/// specified format, then registers it with the DataFusion session context.
///
/// # Arguments
/// * `session` - The DataFusion session context to register the table with
/// * `base_url` - Base URL for the dataset location
/// * `format` - The data format (Parquet, Vortex, etc.) to register
///
/// # Errors
/// Returns an error if table registration fails or if the format is unsupported.
pub async fn register_table(
    session: &SessionContext,
    base_url: &Url,
    format: Format,
) -> Result<()> {
    let table_path = base_url.join(&format!("{}/output4.{}", format.ext(), format.ext()))?;
    let table_url = ListingTableUrl::try_new(table_path, None)?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(
            ListingOptions::new(match format {
                Format::Csv => {
                    return Err(anyhow::anyhow!(
                        "CSV format is not supported for statpopgen benchmark"
                    ));
                }
                Format::Arrow => {
                    return Err(anyhow::anyhow!(
                        "Arrow format is not supported for statpopgen benchmark"
                    ));
                }
                Format::Parquet => Arc::from(ParquetFormat::new()),
                Format::OnDiskVortex => Arc::from(VortexFormat::new(SESSION.clone())),
                Format::VortexCompact => Arc::from(VortexFormat::new(SESSION.clone())),
                Format::OnDiskDuckDB => {
                    return Err(anyhow::anyhow!(
                        "DuckDB format should not be registered through DataFusion"
                    ));
                }
                #[cfg(feature = "lance")]
                Format::Lance => unimplemented!(),
            })
            .with_session_config_options(session.state().config()),
        )
        .infer_schema(&session.state())
        .await?;
    let listing_table = Arc::new(ListingTable::try_new(config)?);
    session.register_table("statpopgen", listing_table)?;
    Ok(())
}

impl Benchmark for StatPopGenBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        let queries_file = Path::new(env!("CARGO_MANIFEST_DIR"))
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

    fn generate_data(&self, target: &Target) -> Result<()> {
        match self.data_url.scheme() {
            "file" => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(self.download_parquet())?;
                match target.format() {
                    Format::Csv => anyhow::bail!("CSV format is not supported by statpopgen"),
                    Format::Arrow => anyhow::bail!("Arrow format is not supported by statpopgen"),
                    Format::Parquet => {}
                    Format::OnDiskVortex | Format::VortexCompact => {
                        rt.block_on(self.parquet_to_vortex(target.format()))?
                    }
                    Format::OnDiskDuckDB => {
                        // We wait to do this until register_tables because we don't have a duckdb
                        // instance until then.
                        //
                        // DuckDBCtx::register_tables will automatically rewrite the Parquet file
                        // into a duckdb file.
                    }
                    #[cfg(feature = "lance")]
                    Format::Lance => unimplemented!(),
                }
                Ok(())
            }
            scheme => anyhow::bail!(
                "Unsupported URL scheme '{}' - only 'file://' URLs are supported for local datasets",
                scheme
            ),
        }
    }

    #[allow(async_fn_in_trait)]
    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> Result<()> {
        let dataset = self.dataset();

        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                dataset
                    .register_tables(&ctx.session, &self.data_url, format)
                    .await
            }
            EngineCtx::DuckDB(ctx) => ctx.register_tables(
                &self.data_url.join(&(self.n_rows.to_string() + "/"))?,
                format,
                &dataset,
            ),
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::StatPopGen {
            n_rows: self.n_rows,
        }
    }

    fn expected_row_counts(&self) -> Option<&[usize]> {
        // Statpopgen reference row counts
        self.expected_row_counts.as_ref().map(|x| x.as_ref())
    }

    // Dataset-specific methods (inlined from BenchmarkDataset)

    fn dataset_name(&self) -> &str {
        "statpopgen"
    }

    fn dataset_display(&self) -> String {
        "statpopgen".to_string()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }
}
