use std::fmt::Display;

use anyhow::Result;
use async_trait::async_trait;
use datafusion::prelude::SessionContext;
use url::Url;
use vortex::ArrayRef;

use crate::{Format, clickbench};

pub mod data_downloads;
pub mod file;
pub mod struct_list_of_ints;
pub mod taxi_data;
pub mod tpch_l_comment;

#[async_trait]
pub trait Dataset {
    fn name(&self) -> &str;

    async fn to_vortex_array(&self) -> ArrayRef;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkDataset {
    TpcH,
    TpcDS,
    ClickBench { single_file: bool },
}

impl Display for BenchmarkDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkDataset::TpcH => write!(f, "tpch"),
            BenchmarkDataset::TpcDS => write!(f, "tpcds"),
            BenchmarkDataset::ClickBench { single_file } => {
                if *single_file {
                    write!(f, "clickbench-single")
                } else {
                    write!(f, "clickbench-partitioned")
                }
            }
        }
    }
}

impl BenchmarkDataset {
    pub fn tables(&self) -> &[&'static str] {
        match self {
            BenchmarkDataset::TpcDS => &[
                "call_center",
                "catalog_sales",
                "customer_demographics",
                "income_band",
                "store_returns",
                "warehouse",
                "web_sales",
                "catalog_page",
                "customer",
                "date_dim",
                "inventory",
                "promotion",
                "ship_mode",
                "store_sales",
                "web_page",
                "web_site",
                "catalog_returns",
                "customer_address",
                "household_demographics",
                "item",
                "reason",
                "store",
                "time_dim",
                "web_returns",
            ],

            BenchmarkDataset::TpcH => &[
                "customer", "lineitem", "nation", "orders", "part", "partsupp", "region",
                "supplier",
            ],

            BenchmarkDataset::ClickBench { .. } => todo!(),
        }
    }

    pub fn parquet_path(&self, base_url: &Url) -> Result<Url> {
        Ok(base_url.join("parquet/")?)
    }

    pub fn vortex_path(&self, base_url: &Url) -> Result<Url> {
        match self {
            BenchmarkDataset::TpcH | BenchmarkDataset::TpcDS => {
                Ok(base_url.join(&format!("{}/", Format::OnDiskVortex))?)
            }
            BenchmarkDataset::ClickBench { .. } => {
                // ClickBench vortex files are stored in "vortex/" subdirectory
                Ok(base_url.join("vortex/")?)
            }
        }
    }

    pub async fn register_tables(
        &self,
        session: &SessionContext,
        base_url: &Url,
        format: Format,
    ) -> Result<()> {
        // Register tables synchronously to avoid nested runtime issues
        match (self, format) {
            (BenchmarkDataset::TpcH, _) | (BenchmarkDataset::TpcDS, _) => {
                // TPC-H tables are handled separately
            }
            (BenchmarkDataset::ClickBench { single_file }, Format::Parquet) => {
                clickbench::register_parquet_files(
                    session,
                    "hits",
                    base_url,
                    &clickbench::HITS_SCHEMA,
                    *single_file,
                )?;
            }
            (BenchmarkDataset::ClickBench { single_file }, Format::OnDiskVortex) => {
                clickbench::register_vortex_files(
                    session.clone(),
                    "hits",
                    base_url,
                    Some(clickbench::HITS_SCHEMA.clone()),
                    *single_file,
                )
                .await?;
            }
            (BenchmarkDataset::ClickBench { .. }, _) => {
                anyhow::bail!("Unsupported format for ClickBench: {}", format);
            }
        }

        Ok(())
    }
}
