// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use vortex::array::ArrayRef;

use crate::clickbench::Flavor;

pub mod data_downloads;
pub mod feature_vectors;
pub mod nested_lists;
pub mod nested_structs;
pub mod struct_list_of_ints;
pub mod taxi_data;
pub mod tpch_l_comment;

pub(crate) const DEFAULT_BENCHMARK_RUNNER_ID: &str = "unknown";

#[async_trait]
pub trait Dataset {
    fn name(&self) -> &str;

    async fn to_vortex_array(&self) -> Result<ArrayRef>;

    /// Get the path to the parquet file for this dataset.
    ///
    /// This method ensures the parquet file exists (downloading if necessary)
    /// and returns the path to it.
    async fn to_parquet_path(&self) -> Result<PathBuf>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkDataset {
    #[serde(rename = "tpch")]
    TpcH { scale_factor: String },
    #[serde(rename = "tpcds")]
    TpcDS { scale_factor: String },
    #[serde(rename = "clickbench")]
    ClickBench { flavor: Flavor },
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
            BenchmarkDataset::TpcH { .. } => "tpch",
            BenchmarkDataset::TpcDS { .. } => "tpcds",
            BenchmarkDataset::ClickBench { .. } => "clickbench",
            BenchmarkDataset::PublicBi { .. } => "public-bi",
            BenchmarkDataset::StatPopGen { .. } => "statpopgen",
            BenchmarkDataset::PolarSignals { .. } => "polarsignals",
            BenchmarkDataset::Fineweb => "fineweb",
            BenchmarkDataset::GhArchive => "gharchive",
        }
    }

    /// Return the globally unique path prefix used for query benchmark result IDs.
    pub fn benchmark_id_path(&self, benchmark_runner: &str, query_idx: usize) -> String {
        let runner_id = normalize_benchmark_runner_id(benchmark_runner);
        let query_segment = format!("q{query_idx:02}");
        let dataset_path = match self {
            BenchmarkDataset::TpcH { scale_factor } => {
                format!(
                    "tpch/sf_{}/{query_segment}",
                    scale_factor_slug(scale_factor)
                )
            }
            BenchmarkDataset::TpcDS { scale_factor } => {
                format!(
                    "tpcds/sf_{}/{query_segment}",
                    scale_factor_slug(scale_factor)
                )
            }
            BenchmarkDataset::ClickBench { flavor } => {
                format!(
                    "clickbench/flavor_{}/{query_segment}",
                    slug(&flavor.to_string())
                )
            }
            BenchmarkDataset::PublicBi { name } => {
                format!("public-bi/{}/{query_segment}", slug(name))
            }
            BenchmarkDataset::StatPopGen { n_rows } => {
                format!("statpopgen/rows_{n_rows}/{query_segment}")
            }
            BenchmarkDataset::PolarSignals { n_rows } => {
                format!("polarsignals/rows_{n_rows}/{query_segment}")
            }
            BenchmarkDataset::Fineweb => format!("fineweb/{query_segment}"),
            BenchmarkDataset::GhArchive => format!("gharchive/{query_segment}"),
        };
        format!("{dataset_path}/{runner_id}")
    }

    pub fn benchmark_memory_id_path(&self, benchmark_runner: &str, query_idx: usize) -> String {
        format!(
            "memory/{}",
            self.benchmark_id_path(benchmark_runner, query_idx)
        )
    }
}

pub(crate) fn normalize_benchmark_runner_id(benchmark_runner: &str) -> String {
    let benchmark_runner = benchmark_runner.trim().replace('/', "_");
    if benchmark_runner.is_empty() {
        DEFAULT_BENCHMARK_RUNNER_ID.to_string()
    } else {
        benchmark_runner
    }
}

fn scale_factor_slug(scale_factor: &str) -> String {
    slug(scale_factor.strip_suffix(".0").unwrap_or(scale_factor))
}

fn slug(value: &str) -> String {
    let mut slug = String::new();
    let mut last_was_separator = false;

    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_separator = false;
        } else if !last_was_separator && !slug.is_empty() {
            slug.push('_');
            last_was_separator = true;
        }
    }

    if slug.ends_with('_') {
        slug.pop();
    }

    slug
}

impl Display for BenchmarkDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkDataset::TpcH { scale_factor } => write!(f, "tpch(sf={scale_factor})"),
            BenchmarkDataset::TpcDS { scale_factor } => write!(f, "tpcds(sf={scale_factor})"),
            BenchmarkDataset::ClickBench { flavor, .. } => match flavor {
                Flavor::Partitioned => write!(f, "clickbench-partitioned"),
                Flavor::Single => write!(f, "clickbench-single"),
            },
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

impl BenchmarkDataset {
    pub fn tables(&self) -> &[&'static str] {
        match self {
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
            BenchmarkDataset::ClickBench { .. } | BenchmarkDataset::PublicBi { .. } => todo!(),
            BenchmarkDataset::StatPopGen { .. } => &["statpopgen"],
            BenchmarkDataset::PolarSignals { .. } => &["stacktraces"],
            BenchmarkDataset::Fineweb => &["fineweb"],
            BenchmarkDataset::GhArchive => &["events"],
        }
    }
}
