// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS benchmark implementation

use anyhow::{Result, anyhow};
use datafusion::prelude::SessionContext;
use log::info;
use url::Url;

use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::helpers::urls::benchmark_data_url;
use crate::tpcds::duckdb::generate_tpcds;
use crate::tpcds::tpcds_queries;
use crate::{BenchmarkDataset, Format, Target};

/// TPC-DS benchmark implementation
pub struct TpcDsBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl TpcDsBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            scale_factor: scale_factor.clone(),
            data_url: Self::create_data_url(&use_remote_data_dir, &scale_factor)?,
        })
    }

    fn create_data_url(remote_data_dir: &Option<String>, scale_factor: &str) -> Result<Url> {
        benchmark_data_url("tpcds", Some(scale_factor), remote_data_dir)
    }
}

impl Benchmark for TpcDsBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpcds_queries().collect())
    }

    fn generate_data(&self, target: &Target) -> Result<()> {
        // Use connection-based TPC-DS data generation
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

        info!(
            "Generating TPC-DS data with scale factor {} for format {:?}",
            self.scale_factor,
            target.format()
        );

        generate_tpcds(base_data_dir, self.scale_factor.clone(), target.format())?;
        Ok(())
    }

    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> Result<()> {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                // Register TPC-DS tables similar to how TPC-H does it
                self.register_tpcds_tables(&ctx.session, &self.data_url, format)
                    .await
            }
            EngineCtx::DuckDB(ctx) => {
                ctx.register_tables(&self.data_url, format, &self.dataset())?;
                Ok(())
            }
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcDS {
            scale_factor: self.scale_factor.clone(),
        }
    }

    fn dataset_name(&self) -> &str {
        "tpcds"
    }

    fn dataset_display(&self) -> String {
        format!("tpcds(sf={})", self.scale_factor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn expected_row_counts(&self) -> Option<&[usize]> {
        None
    }

    fn tables(&self) -> &[&'static str] {
        // TPC-DS has 24 tables
        &[
            "call_center",
            "catalog_page",
            "catalog_returns",
            "catalog_sales",
            "customer",
            "customer_address",
            "customer_demographics",
            "date_dim",
            "household_demographics",
            "income_band",
            "inventory",
            "item",
            "promotion",
            "reason",
            "ship_mode",
            "store",
            "store_returns",
            "store_sales",
            "time_dim",
            "warehouse",
            "web_page",
            "web_returns",
            "web_sales",
            "web_site",
        ]
    }

    fn validate_result(&self, _queries: Vec<usize>) -> Result<()> {
        Ok(())
    }
}

impl TpcDsBenchmark {
    async fn register_tpcds_tables(
        &self,
        session: &SessionContext,
        base_dir: &Url,
        format: Format,
    ) -> Result<()> {
        use crate::datasets::configs::TpcDsDataset;
        use crate::datasets::unified_registration::register_dataset_tables;

        let dataset = TpcDsDataset {
            scale_factor: self.scale_factor.clone(),
        };

        register_dataset_tables(session, &dataset, base_dir, format).await
    }
}
