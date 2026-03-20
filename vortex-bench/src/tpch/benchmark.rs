// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPCH benchmark implementation

use anyhow::anyhow;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::FilePattern;
use crate::QuerySource;
use crate::resolve_data_url;
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
use crate::tpch::tpchgen;
use crate::tpch::tpchgen::TpchGenOptions;

/// TPCH benchmark implementation
pub struct TpcHBenchmark {
    descriptor: BenchmarkDescriptor,
    pub scale_factor: String,
}

impl TpcHBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = resolve_data_url(
            &format!("tpch/{scale_factor}"),
            use_remote_data_dir.as_deref(),
        )?;

        let expected_row_counts = match scale_factor.as_str() {
            "1.0" => Some(EXPECTED_ROW_COUNTS_SF1.to_vec()),
            "10.0" => Some(EXPECTED_ROW_COUNTS_SF10.to_vec()),
            _ => None,
        };

        let mut desc = BenchmarkDescriptor::new(
            "tpch",
            data_url,
            BenchmarkDataset::TpcH {
                scale_factor: scale_factor.clone(),
            },
        )
        .with_display(format!("tpch(sf={scale_factor})"))
        .with_queries(QuerySource::numbered_q("tpch", 1, 22))
        .with_table("customer", Some(CUSTOMER.clone()))
        .with_table("lineitem", Some(LINEITEM.clone()))
        .with_table("nation", Some(NATION.clone()))
        .with_table("orders", Some(ORDERS.clone()))
        .with_table("part", Some(PART.clone()))
        .with_table("partsupp", Some(PARTSUPP.clone()))
        .with_table("region", Some(REGION.clone()))
        .with_table("supplier", Some(SUPPLIER.clone()))
        .with_file_pattern(FilePattern::TablePrefix);

        if let Some(counts) = expected_row_counts {
            desc = desc.with_expected_row_counts(counts);
        }

        Ok(Self {
            descriptor: desc,
            scale_factor,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for TpcHBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.descriptor.data_url.scheme() != "file" {
            return Ok(());
        }

        let base_data_dir = self
            .descriptor
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.descriptor.data_url.as_str()))?;

        let options = TpchGenOptions::new(self.scale_factor.clone(), base_data_dir)
            .with_max_file_size_mb(Some(600));

        tpchgen::generate_tpch_tables(options).await?;

        Ok(())
    }
}
