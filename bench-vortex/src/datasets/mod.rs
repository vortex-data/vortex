// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use anyhow::Result;
use async_trait::async_trait;
use datafusion::prelude::SessionContext;
use serde::Serialize;
use url::Url;
use vortex::array::ArrayRef;

use crate::Format;
use crate::clickbench;
use crate::clickbench::Flavor;
#[cfg(feature = "lance")]
use crate::file::register_lance_files;
use crate::fineweb;
use crate::realnest::gharchive;
use crate::statpopgen;

pub mod data_downloads;
pub mod file;
pub mod struct_list_of_ints;
pub mod taxi_data;
pub mod tpch_l_comment;

#[async_trait]
pub trait Dataset {
    fn name(&self) -> &str;

    async fn to_vortex_array(&self) -> Result<ArrayRef>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
            BenchmarkDataset::Fineweb => "fineweb",
            BenchmarkDataset::GhArchive => "gharchive",
        }
    }
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
            BenchmarkDataset::Fineweb => &["fineweb"],
            BenchmarkDataset::GhArchive => &["events"],
        }
    }

    pub fn format_path(&self, format: Format, base_url: &Url) -> Result<Url> {
        Ok(base_url.join(&format!("{format}/"))?)
    }

    pub async fn register_tables(
        &self,
        session: &SessionContext,
        base_url: &Url,
        format: Format,
    ) -> Result<()> {
        match (self, format) {
            (BenchmarkDataset::TpcH { .. }, _) | (BenchmarkDataset::TpcDS { .. }, _) => {
                // TPC-H tables are handled separately
            }
            (BenchmarkDataset::ClickBench { .. }, Format::Parquet) => {
                clickbench::register_parquet_files(
                    session,
                    "hits",
                    base_url,
                    &clickbench::HITS_SCHEMA,
                    Some(glob::Pattern::new("*.parquet")?),
                )?;
            }
            (BenchmarkDataset::ClickBench { .. }, Format::OnDiskVortex | Format::VortexCompact) => {
                clickbench::register_vortex_files(
                    session.clone(),
                    "hits",
                    base_url,
                    Some(clickbench::HITS_SCHEMA.clone()),
                    Some(glob::Pattern::new("*.vortex")?),
                )
                .await?;
            }
            #[cfg(feature = "lance")]
            (cb @ BenchmarkDataset::ClickBench { .. }, Format::Lance) => {
                register_lance_files(session, "hits", base_url, cb).await?;
            }
            (BenchmarkDataset::ClickBench { .. }, _) => {
                anyhow::bail!("Unsupported format for ClickBench: {}", format);
            }
            (BenchmarkDataset::PublicBi { .. }, _) => {
                anyhow::bail!("public bi unsupported for now")
            }
            (BenchmarkDataset::StatPopGen { .. }, Format::Parquet) => {
                statpopgen::register_table(session, base_url, Format::Parquet).await?
            }
            (BenchmarkDataset::StatPopGen { .. }, Format::OnDiskVortex) => {
                statpopgen::register_table(session, base_url, Format::OnDiskVortex).await?
            }
            (BenchmarkDataset::StatPopGen { .. }, format) => {
                anyhow::bail!("StatPopGen in {format} unsupported in DataFusion")
            }
            (BenchmarkDataset::Fineweb, format) => {
                fineweb::register_table(session, base_url, format).await?
            }
            (BenchmarkDataset::GhArchive, format) => {
                gharchive::register_table(session, base_url, format).await?
            }
        }

        Ok(())
    }
}
