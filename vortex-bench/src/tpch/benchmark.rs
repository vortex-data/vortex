// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPCH benchmark implementation

use glob::Pattern;
use tracing::info;
use tracing::warn;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::tpch::EXPECTED_ROW_COUNTS_SF1;
use crate::tpch::EXPECTED_ROW_COUNTS_SF10;
use crate::tpch::schema::CUSTOMER;
use crate::tpch::schema::LINEITEM;
use crate::tpch::schema::NATION;
use crate::tpch::schema::ORDERS;
use crate::tpch::schema::PART;
use crate::tpch::schema::PARTSUPP;
use crate::tpch::schema::REGION;
use crate::tpch::schema::SUPPLIER;
use crate::tpch::tpch_queries;
use crate::tpch::tpchgen;
use crate::tpch::tpchgen::TpchGenOptions;

/// TPCH benchmark implementation
pub struct TpcHBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl TpcHBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            scale_factor: scale_factor.clone(),
            data_url: Self::create_data_url(use_remote_data_dir.as_deref(), &scale_factor)?,
        })
    }

    fn create_data_url(remote_data_dir: Option<&str>, scale_factor: &str) -> anyhow::Result<Url> {
        match remote_data_dir {
            None => {
                let data_dir = "tpch".to_data_path();
                let data_dir_with_sf = data_dir.join(scale_factor);
                Url::from_directory_path(&data_dir_with_sf).map_err(|_| {
                    anyhow::anyhow!(
                        "Failed to create URL from directory path: {:?}",
                        &data_dir_with_sf
                    )
                })
            }
            Some(remote_data_dir) => {
                if !remote_data_dir.ends_with("/") {
                    warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                    );
                }
                info!(
                    concat!(
                        "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                        "If it does not, you should kill this command, locally generate the files (by running without\n",
                        "--use-remote-data-dir) and upload data/tpch/{}/ to some remote location.",
                    ),
                    remote_data_dir, scale_factor,
                );
                Ok(Url::parse(remote_data_dir)?)
            }
        }
    }
}

#[async_trait::async_trait]
impl Benchmark for TpcHBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(tpch_queries().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url.as_str()))?;

        let options = TpchGenOptions::new(self.scale_factor.clone(), base_data_dir)
            .with_max_file_size_mb(Some(600));

        tpchgen::generate_tpch_tables(options).await?;

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        match self.scale_factor.as_str() {
            "1.0" => Some(EXPECTED_ROW_COUNTS_SF1.to_vec()),
            "10.0" => Some(EXPECTED_ROW_COUNTS_SF10.to_vec()),
            _ => None, // Unsupported scale factor
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcH {
            scale_factor: self.scale_factor.clone(),
        }
    }

    fn dataset_name(&self) -> &str {
        "tpch"
    }

    fn dataset_display(&self) -> String {
        format!("tpch(sf={})", self.scale_factor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![
            TableSpec::new("customer", Some(CUSTOMER.clone())),
            TableSpec::new("lineitem", Some(LINEITEM.clone())),
            TableSpec::new("nation", Some(NATION.clone())),
            TableSpec::new("orders", Some(ORDERS.clone())),
            TableSpec::new("part", Some(PART.clone())),
            TableSpec::new("partsupp", Some(PARTSUPP.clone())),
            TableSpec::new("region", Some(REGION.clone())),
            TableSpec::new("supplier", Some(SUPPLIER.clone())),
        ]
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            format!("{}_*.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
