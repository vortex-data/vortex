// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS benchmark implementation

use std::fs;

use anyhow::{Result, anyhow};
use log::info;
use url::Url;

use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::engines::ddb::DuckDBCtx;
use crate::tpcds::tpcds_queries;
use crate::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use crate::{BenchmarkDataset, Format, IdempotentPath, Target};

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
        match remote_data_dir {
            None => {
                let data_dir = "tpcds".to_data_path();
                let data_dir_with_sf = data_dir.join(scale_factor);
                Url::from_directory_path(&data_dir_with_sf).map_err(|_| {
                    anyhow!(
                        "Failed to create URL from directory path: {:?}",
                        &data_dir_with_sf
                    )
                })
            }
            Some(remote_data_dir) => {
                let mut url = Url::parse(remote_data_dir)?;
                if !url.path().ends_with('/') {
                    url.set_path(&format!("{}/", url.path()));
                }
                Ok(url)
            }
        }
    }
}

impl Benchmark for TpcDsBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpcds_queries().collect())
    }

    fn generate_data(&self, target: &Target) -> Result<()> {
        // TOD: move to tpchgen-rs when it supports TPC-DS
        match target.format() {
            Format::OnDiskDuckDB => {
                // Use DuckDB's dsdgen function to generate TPC-DS data
                let base_data_dir = self
                    .data_url
                    .to_file_path()
                    .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

                // Create output directory
                fs::create_dir_all(&base_data_dir)?;

                // Generate TPC-DS data using DuckDB
                let duckdb_ctx = DuckDBCtx::new(self.dataset(), target.format())?;

                // Install and load the tpcds extension
                let _result1 = duckdb_ctx.execute_query("INSTALL tpcds")?;
                let _result2 = duckdb_ctx.execute_query("LOAD tpcds")?;

                // Generate data using dsdgen
                let generate_sql = format!("CALL dsdgen(sf={});", self.scale_factor);

                info!(
                    "Generating TPC-DS data with scale factor {}",
                    self.scale_factor
                );
                let _result = duckdb_ctx.execute_query(&generate_sql)?;

                Ok(())
            }
            _ => {
                // Use the shared TPC data generation function
                let base_data_dir = self
                    .data_url
                    .to_file_path()
                    .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

                let opts = DuckdbTpcOptions::new(base_data_dir, TpcDataset::TpcDs, target.format())
                    .with_scale_factor(self.scale_factor.clone());

                info!(
                    "Generating TPC-DS data with scale factor {} for format {:?}",
                    self.scale_factor,
                    target.format()
                );

                generate_tpc(opts)?;
                Ok(())
            }
        }
    }

    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> Result<()> {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                let dataset = self.dataset();
                dataset
                    .register_tables(&ctx.session, &self.data_url, format)
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
