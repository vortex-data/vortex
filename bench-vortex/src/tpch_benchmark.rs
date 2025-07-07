// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPCH benchmark implementation

use anyhow::Result;
use url::Url;
use log::{info, warn};

use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use crate::tpch::{EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, tpch_queries};
use crate::{BenchmarkDataset, Format, Target, IdempotentPath};
use vortex::error::VortexExpect;

/// TPCH benchmark implementation
pub struct TpcHBenchmark {
    pub scale_factor: u32,
    pub use_remote_data_dir: Option<String>,
}

impl TpcHBenchmark {
    pub fn new(scale_factor: u32, use_remote_data_dir: Option<String>) -> Self {
        Self {
            scale_factor,
            use_remote_data_dir,
        }
    }
}

impl Benchmark for TpcHBenchmark {
    fn get_queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpch_queries().collect())
    }

    fn generate_data(&self, _data_url: &Url, target: &Target) -> Result<()> {
        match &self.use_remote_data_dir {
            None => {
                // Generate data for the specific target format (idempotent)
                let format = if target.format() == Format::Arrow {
                    Format::Csv
                } else {
                    target.format()
                };
                
                let opts = DuckdbTpcOptions::new("tpch".to_data_path(), TpcDataset::TpcH, format)
                    .with_scale_factor(self.scale_factor);
                generate_tpc(opts)?;

                let data_dir = "tpch".to_data_path();
                let data_dir = data_dir.to_str().vortex_expect("path must be utf8");
                info!("Generated or verified TPCH data for format {} at {data_dir}.", format);
                Ok(())
            }
            Some(tpch_benchmark_remote_data_dir) => {
                if !tpch_benchmark_remote_data_dir.ends_with("/") {
                    warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                    );
                }
                info!(
                    concat!(
                        "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                        "If it does not, you should kill this command, locally generate the files (by running without\n",
                        "--use-remote-data-dir) and upload data/tpch/1/ to some remote location.",
                    ),
                    tpch_benchmark_remote_data_dir,
                );
                Ok(())
            }
        }
    }

    #[allow(async_fn_in_trait)]
    async fn register_tables(
        &self,
        engine_ctx: &EngineCtx,
        data_url: &Url,
        format: Format,
    ) -> Result<()> {
        let dataset = self.get_dataset();

        match engine_ctx {
            EngineCtx::DataFusion(_ctx) => {
                // Tables are already registered by load_datasets in setup_engine_context
                Ok(())
            }
            EngineCtx::DuckDB(ctx) => {
                ctx.register_tables(data_url, format, &dataset)?;
                Ok(())
            }
        }
    }

    fn get_dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcH {
            scale_factor: self.scale_factor,
        }
    }

    fn get_expected_row_counts(&self) -> Option<&[usize]> {
        match self.scale_factor {
            1 => Some(&EXPECTED_ROW_COUNTS_SF1),
            10 => Some(&EXPECTED_ROW_COUNTS_SF10),
            _ => None, // Unsupported scale factor
        }
    }


}