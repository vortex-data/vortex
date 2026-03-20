// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS benchmark implementation

use anyhow::Result;
use anyhow::anyhow;
use tracing::info;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::FilePattern;
use crate::Format;
use crate::QuerySource;
use crate::resolve_data_url;
use crate::tpcds::duckdb::generate_tpcds;

/// TPC-DS benchmark implementation
pub struct TpcDsBenchmark {
    descriptor: BenchmarkDescriptor,
    pub scale_factor: String,
}

impl TpcDsBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> Result<Self> {
        let data_url = resolve_data_url(
            &format!("tpcds/{scale_factor}"),
            use_remote_data_dir.as_deref(),
        )?;

        let desc = BenchmarkDescriptor::new(
            "tpcds",
            data_url,
            BenchmarkDataset::TpcDS {
                scale_factor: scale_factor.clone(),
            },
        )
        .with_display(format!("tpcds(sf={scale_factor})"))
        .with_queries(QuerySource::numbered_zero_padded("tpcds", 1, 99))
        .with_table("call_center", None)
        .with_table("catalog_page", None)
        .with_table("catalog_returns", None)
        .with_table("catalog_sales", None)
        .with_table("customer", None)
        .with_table("customer_address", None)
        .with_table("customer_demographics", None)
        .with_table("date_dim", None)
        .with_table("household_demographics", None)
        .with_table("income_band", None)
        .with_table("inventory", None)
        .with_table("item", None)
        .with_table("promotion", None)
        .with_table("reason", None)
        .with_table("ship_mode", None)
        .with_table("store", None)
        .with_table("store_returns", None)
        .with_table("store_sales", None)
        .with_table("time_dim", None)
        .with_table("warehouse", None)
        .with_table("web_page", None)
        .with_table("web_returns", None)
        .with_table("web_sales", None)
        .with_table("web_site", None)
        .with_file_pattern(FilePattern::TableExact);

        Ok(Self {
            descriptor: desc,
            scale_factor,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for TpcDsBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> Result<()> {
        let base_data_dir = self
            .descriptor
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.descriptor.data_url))?;

        info!(
            "Generating TPC-DS data with scale factor {} for format {:?}",
            self.scale_factor,
            Format::Parquet
        );

        generate_tpcds(base_data_dir, self.scale_factor.clone())?;

        Ok(())
    }
}
