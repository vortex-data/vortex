// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data acquisition and table specs for SQLStorm origins.
//!
//! `table_names` is the single source of truth for each origin's table list;
//! both `table_specs` (used by `SqlstormBenchmark`) and
//! `BenchmarkDataset::tables()` (used by the registration layer) delegate here.

use url::Url;

use crate::TableSpec;
use crate::sqlstorm::SqlstormOrigin;

/// Table names per origin (single source of truth).
///
/// TPC-H and TPC-DS mirror the corresponding benchmark's table lists.
/// StackOverflow lists the 13 tables from `stackoverflow.dbschema.json`.
/// JOB (IMDB) is populated in a later task.
pub fn table_names(origin: SqlstormOrigin) -> &'static [&'static str] {
    match origin {
        SqlstormOrigin::TpcH => &[
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ],
        SqlstormOrigin::TpcDs => &[
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
        ],
        SqlstormOrigin::StackOverflow => &[
            "posthistorytypes",
            "linktypes",
            "posttypes",
            "closereasontypes",
            "votetypes",
            "users",
            "badges",
            "posts",
            "comments",
            "posthistory",
            "postlinks",
            "tags",
            "votes",
        ],
        SqlstormOrigin::Job => &[],
    }
}

/// Table specs for an origin (schema inferred at registration time — `None`).
pub fn table_specs(origin: SqlstormOrigin) -> Vec<TableSpec> {
    table_names(origin)
        .iter()
        .map(|n| TableSpec::new(n, None))
        .collect()
}

/// Download and convert StackOverflow `dba` data to Parquet. Implemented in a later task.
pub async fn generate_stackoverflow(_data_url: &Url) -> anyhow::Result<()> {
    anyhow::bail!("stackoverflow data-gen not yet implemented")
}

/// Download and convert IMDB/JOB data to Parquet. Implemented in a later task.
pub async fn generate_job(_data_url: &Url) -> anyhow::Result<()> {
    anyhow::bail!("job data-gen not yet implemented")
}
