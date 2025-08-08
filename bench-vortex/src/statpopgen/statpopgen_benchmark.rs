// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_datafusion::VortexFormat;

use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::statpopgen::schema::SCHEMA;
use crate::{BenchmarkDataset, Format, Target};

/// Statpopgen benchmark implementation
pub struct StatPopGenBenchmark {
    pub data_url: Url,
    pub n_rows: u64,
    pub expected_row_counts: Vec<usize>,
}

impl StatPopGenBenchmark {
    pub fn parquet_path(&self) -> VortexResult<PathBuf> {
        self.data_url
            .join(&(self.n_rows.to_string() + "/parquet/"))?
            .join("gnomad.genomes.v3.1.2.hgdp_tgp.chr21.parquet")?
            .to_file_path()
            .map_err(|_| vortex_err!("data url must be a local file system path"))
    }

    pub fn vortex_path(&self) -> VortexResult<PathBuf> {
        self.data_url
            .join(&(self.n_rows.to_string() + "/vortex-file-compressed/"))?
            .join("gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vortex")?
            .to_file_path()
            .map_err(|_| vortex_err!("data url must be a local file system path"))
    }

    pub fn vortex_compact_path(&self) -> VortexResult<PathBuf> {
        self.data_url
            .join(&(self.n_rows.to_string() + "/vortex-compact/"))?
            .join("gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vortex")?
            .to_file_path()
            .map_err(|_| vortex_err!("data url must be a local file system path"))
    }

    pub fn new(data_url: Url, n_rows: u64) -> VortexResult<Self> {
        // let url = Self::create_data_url(&use_remote_data_dir, flavor)?;
        let n_variants =
            usize::try_from(n_rows).map_err(|_| vortex_err!("too many rows for this machine"))?;
        let expected_row_counts = vec![1, 1, n_variants, n_variants, n_variants];

        Ok(Self {
            data_url,
            n_rows,
            expected_row_counts,
        })
    }
}

pub fn register_table(session: &SessionContext, base_url: &Url, format: Format) -> Result<()> {
    let table_path = base_url.join(&format!("{}/output4.{}", format.ext(), format.ext()))?;
    let table_url = ListingTableUrl::try_new(table_path, None)?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(
            ListingOptions::new(match format {
                Format::Csv => todo!(),
                Format::Arrow => todo!(),
                Format::Parquet => Arc::from(ParquetFormat::new()),
                Format::OnDiskVortex => Arc::from(VortexFormat::default()),
                Format::VortexCompact => todo!(),
                Format::OnDiskDuckDB => todo!(),
            })
            .with_session_config_options(session.state().config()),
        )
        .with_schema(SCHEMA.clone());
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
                }
                Ok(())
            }
            _ => todo!(),
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
        Some(&self.expected_row_counts)
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
