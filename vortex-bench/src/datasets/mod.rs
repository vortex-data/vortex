// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;

use crate::clickbench::Flavor;

pub mod data_downloads;
pub mod feature_vectors;
pub mod nested_lists;
pub mod nested_structs;
pub mod struct_list_of_ints;
pub mod taxi_data;
pub mod tpch_l_comment;

use std::path::PathBuf;

pub(crate) const DEFAULT_BENCHMARK_RUNNER_ID: &str = "unknown";

pub(crate) fn normalize_benchmark_runner_id(benchmark_runner: &str) -> String {
    let benchmark_runner = benchmark_runner.trim().replace('/', "_");
    if benchmark_runner.is_empty() {
        DEFAULT_BENCHMARK_RUNNER_ID.to_string()
    } else {
        benchmark_runner
    }
}

#[async_trait]
pub trait Dataset {
    fn name(&self) -> &str;

    /// Map this dataset to the v3 `(dataset, dataset_variant)` pair emitted
    /// in `compression_*` records.
    ///
    /// Default: `(name(), None)`. Override only when a suite needs a
    /// different dataset name on the wire than its `name()` returns. The
    /// query-side equivalent is documented on
    /// [`crate::v3::benchmark_dataset_dims`].
    fn v3_dataset_dims(&self) -> (&str, Option<&str>) {
        (self.name(), None)
    }

    async fn to_vortex_array(&self, ctx: &mut ExecutionCtx) -> Result<ArrayRef>;

    /// Get the path to the parquet file for this dataset.
    ///
    /// This method ensures the parquet file exists (downloading if necessary)
    /// and returns the path to it.
    async fn to_parquet_path(&self) -> Result<PathBuf>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkDataset {
    #[serde(rename = "appian")]
    Appian,
    #[serde(rename = "tpch")]
    TpcH { scale_factor: String },
    #[serde(rename = "tpcds")]
    TpcDS { scale_factor: String },
    #[serde(rename = "clickbench")]
    ClickBench { flavor: Flavor },
    #[serde(rename = "clickbench-sorted")]
    ClickBenchSorted,
    #[serde(rename = "public-bi")]
    PublicBi { name: String },
    #[serde(rename = "statpopgen")]
    StatPopGen { n_rows: u64 },
    #[serde(rename = "polarsignals")]
    PolarSignals { n_rows: usize },
    #[serde(rename = "fineweb")]
    Fineweb,
    #[serde(rename = "gharchive")]
    GhArchive,
}

impl BenchmarkDataset {
    pub fn name(&self) -> &str {
        match self {
            BenchmarkDataset::Appian => "appian",
            BenchmarkDataset::TpcH { .. } => "tpch",
            BenchmarkDataset::TpcDS { .. } => "tpcds",
            BenchmarkDataset::ClickBench { .. } => "clickbench",
            BenchmarkDataset::ClickBenchSorted => "clickbench-sorted",
            BenchmarkDataset::PublicBi { .. } => "public-bi",
            BenchmarkDataset::StatPopGen { .. } => "statpopgen",
            BenchmarkDataset::PolarSignals { .. } => "polarsignals",
            BenchmarkDataset::Fineweb => "fineweb",
            BenchmarkDataset::GhArchive => "gharchive",
        }
    }
}

impl Display for BenchmarkDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkDataset::Appian => write!(f, "appian"),
            BenchmarkDataset::TpcH { scale_factor } => write!(f, "tpch(sf={scale_factor})"),
            BenchmarkDataset::TpcDS { scale_factor } => write!(f, "tpcds(sf={scale_factor})"),
            BenchmarkDataset::ClickBench { flavor, .. } => match flavor {
                Flavor::Partitioned => write!(f, "clickbench-partitioned"),
                Flavor::Single => write!(f, "clickbench-single"),
            },
            BenchmarkDataset::ClickBenchSorted => write!(f, "clickbench-sorted"),
            BenchmarkDataset::PublicBi { name } => write!(f, "public-bi({name})"),
            BenchmarkDataset::StatPopGen { n_rows } => write!(f, "statpopgen(n_rows={n_rows})"),
            BenchmarkDataset::PolarSignals { n_rows } => {
                write!(f, "polarsignals(n_rows={n_rows})")
            }
            BenchmarkDataset::Fineweb => write!(f, "fineweb"),
            BenchmarkDataset::GhArchive => write!(f, "gharchive"),
        }
    }
}

const APPIAN_TABLES: &[&str] = &[
    "addressview",
    "categoryview",
    "creditcardview",
    "customerview",
    "orderitemnovelty_update",
    "orderitemview",
    "orderview",
    "productview",
    "taxrecordview",
];

impl BenchmarkDataset {
    pub fn tables(&self) -> &[&'static str] {
        match self {
            BenchmarkDataset::Appian => APPIAN_TABLES,
            BenchmarkDataset::TpcDS { .. } => &[
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
            BenchmarkDataset::TpcH { .. } => &[
                "customer", "lineitem", "nation", "orders", "part", "partsupp", "region",
                "supplier",
            ],
            BenchmarkDataset::ClickBench { .. } | BenchmarkDataset::ClickBenchSorted => &["hits"],
            BenchmarkDataset::PublicBi { .. } => todo!(),
            BenchmarkDataset::StatPopGen { .. } => &["statpopgen"],
            BenchmarkDataset::PolarSignals { .. } => &["stacktraces"],
            BenchmarkDataset::Fineweb => &["fineweb"],
            BenchmarkDataset::GhArchive => &["events"],
        }
    }
}
