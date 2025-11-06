// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete dataset configurations
//!
//! This module contains pure data structures for dataset configuration,
//! without any behavior or registration logic.

use std::fmt;

use crate::clickbench::Flavor;
use super::metadata::{DatasetMetadata, TableInfo};

/// TPC-H dataset configuration
#[derive(Debug, Clone)]
pub struct TpcHDataset {
    pub scale_factor: String,
}

impl DatasetMetadata for TpcHDataset {
    fn name(&self) -> &str {
        "tpch"
    }

    fn variant(&self) -> String {
        self.scale_factor.clone()
    }

    fn tables(&self) -> Vec<TableInfo> {
        ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
            .iter()
            .map(|&name| TableInfo::new(name, format!("{}*.parquet", name)))
            .collect()
    }
}

impl fmt::Display for TpcHDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tpch(sf={})", self.scale_factor)
    }
}

/// TPC-DS dataset configuration
#[derive(Debug, Clone)]
pub struct TpcDsDataset {
    pub scale_factor: String,
}

impl DatasetMetadata for TpcDsDataset {
    fn name(&self) -> &str {
        "tpcds"
    }

    fn variant(&self) -> String {
        self.scale_factor.clone()
    }

    fn tables(&self) -> Vec<TableInfo> {
        [
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
        ]
        .iter()
        .map(|&name| TableInfo::new(name, format!("{}*.parquet", name)))
        .collect()
    }
}

impl fmt::Display for TpcDsDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tpcds(sf={})", self.scale_factor)
    }
}

/// ClickBench dataset configuration
#[derive(Debug, Clone)]
pub struct ClickBenchDataset {
    pub flavor: Flavor,
}

impl DatasetMetadata for ClickBenchDataset {
    fn name(&self) -> &str {
        "clickbench"
    }

    fn variant(&self) -> String {
        self.flavor.to_string()
    }

    fn tables(&self) -> Vec<TableInfo> {
        // ClickBench has a single table with schema
        vec![TableInfo::new("hits", "*.parquet")]
    }
}

impl fmt::Display for ClickBenchDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.flavor {
            Flavor::Partitioned => write!(f, "clickbench-partitioned"),
            Flavor::Single => write!(f, "clickbench-single"),
        }
    }
}

/// Public BI dataset configuration
#[derive(Debug, Clone)]
pub struct PublicBiDataset {
    pub name: String,
}

impl DatasetMetadata for PublicBiDataset {
    fn name(&self) -> &str {
        "public-bi"
    }

    fn variant(&self) -> String {
        self.name.clone()
    }

    fn tables(&self) -> Vec<TableInfo> {
        // Public BI tables vary by dataset
        vec![TableInfo::new(&self.name, format!("{}*.parquet", self.name))]
    }
}

impl fmt::Display for PublicBiDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "public-bi({})", self.name)
    }
}

/// StatPopGen dataset configuration
#[derive(Debug, Clone)]
pub struct StatPopGenDataset {
    pub n_rows: u64,
}

impl DatasetMetadata for StatPopGenDataset {
    fn name(&self) -> &str {
        "statpopgen"
    }

    fn variant(&self) -> String {
        format!("{}", self.n_rows)
    }

    fn tables(&self) -> Vec<TableInfo> {
        vec![TableInfo::new("statpopgen", "*.parquet")]
    }
}

impl fmt::Display for StatPopGenDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "statpopgen(n_rows={})", self.n_rows)
    }
}

/// FineWeb dataset configuration
#[derive(Debug, Clone)]
pub struct FineWebDataset;

impl DatasetMetadata for FineWebDataset {
    fn name(&self) -> &str {
        "fineweb"
    }

    fn tables(&self) -> Vec<TableInfo> {
        vec![TableInfo::new("fineweb", "*.parquet")]
    }
}

impl fmt::Display for FineWebDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fineweb")
    }
}

/// GH Archive dataset configuration
#[derive(Debug, Clone)]
pub struct GhArchiveDataset;

impl DatasetMetadata for GhArchiveDataset {
    fn name(&self) -> &str {
        "gharchive"
    }

    fn tables(&self) -> Vec<TableInfo> {
        vec![TableInfo::new("events", "*.parquet")]
    }
}

impl fmt::Display for GhArchiveDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gharchive")
    }
}