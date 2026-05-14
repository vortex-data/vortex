// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS benchmark implementation

use anyhow::Result;
use anyhow::anyhow;
use glob::Pattern;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::tpcds::duckdb::generate_tpcds;
use crate::tpcds::tpcds_queries;

/// TPC-DS benchmark implementation
pub struct TpcDsBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl TpcDsBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            scale_factor: scale_factor.clone(),
            data_url: Self::create_data_url(use_remote_data_dir.as_deref(), &scale_factor)?,
        })
    }

    fn create_data_url(remote_data_dir: Option<&str>, scale_factor: &str) -> Result<Url> {
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

#[async_trait::async_trait]
impl Benchmark for TpcDsBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpcds_queries().collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

        info!(
            "Generating TPC-DS data with scale factor {} for format {:?}",
            self.scale_factor,
            Format::Parquet
        );

        generate_tpcds(base_data_dir, self.scale_factor.clone())?;

        Ok(())
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

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![
            TableSpec::new("call_center", None),
            TableSpec::new("catalog_page", None),
            TableSpec::new("catalog_returns", None),
            TableSpec::new("catalog_sales", None),
            TableSpec::new("customer", None),
            TableSpec::new("customer_address", None),
            TableSpec::new("customer_demographics", None),
            TableSpec::new("date_dim", None),
            TableSpec::new("household_demographics", None),
            TableSpec::new("income_band", None),
            TableSpec::new("inventory", None),
            TableSpec::new("item", None),
            TableSpec::new("promotion", None),
            TableSpec::new("reason", None),
            TableSpec::new("ship_mode", None),
            TableSpec::new("store", None),
            TableSpec::new("store_returns", None),
            TableSpec::new("store_sales", None),
            TableSpec::new("time_dim", None),
            TableSpec::new("warehouse", None),
            TableSpec::new("web_page", None),
            TableSpec::new("web_returns", None),
            TableSpec::new("web_sales", None),
            TableSpec::new("web_site", None),
        ]
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            format!("{}.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
