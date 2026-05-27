// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQLStorm benchmark: a TPC-DS-shaped suite over a vendored, confirmed-working
//! sample of SQLStorm queries (25 per origin). See the design doc for rationale.

use std::fs;
use std::path::Path;
use std::str::FromStr;

use clap::ValueEnum;

pub mod data;
pub mod row_counts;
pub mod sqlstorm_benchmark;

pub use sqlstorm_benchmark::SqlstormBenchmark;

/// The four SQLStorm base datasets ("origins").
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SqlstormOrigin {
    #[clap(name = "stackoverflow")]
    StackOverflow,
    #[clap(name = "job")]
    Job,
    #[clap(name = "tpch")]
    TpcH,
    #[clap(name = "tpcds")]
    TpcDs,
}

impl SqlstormOrigin {
    /// Stable lowercase identifier; also the vendored-queries subdirectory name.
    pub fn name(self) -> &'static str {
        match self {
            SqlstormOrigin::StackOverflow => "stackoverflow",
            SqlstormOrigin::Job => "job",
            SqlstormOrigin::TpcH => "tpch",
            SqlstormOrigin::TpcDs => "tpcds",
        }
    }

    /// Parse an origin from its `name()` string. Returns `None` for unknown names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "stackoverflow" => Some(SqlstormOrigin::StackOverflow),
            "job" => Some(SqlstormOrigin::Job),
            "tpch" => Some(SqlstormOrigin::TpcH),
            "tpcds" => Some(SqlstormOrigin::TpcDs),
            _ => None,
        }
    }

    /// All four origins in canonical order.
    pub fn all() -> [SqlstormOrigin; 4] {
        [
            SqlstormOrigin::StackOverflow,
            SqlstormOrigin::Job,
            SqlstormOrigin::TpcH,
            SqlstormOrigin::TpcDs,
        ]
    }
}

impl FromStr for SqlstormOrigin {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown sqlstorm origin: {s:?}; valid values are stackoverflow, job, tpch, tpcds"
            )
        })
    }
}

/// Load the vendored, confirmed-working queries for an origin from
/// `vortex-bench/sqlstorm/<origin>/*.sql`, sorted by query id for stable ordering.
pub fn sqlstorm_queries(origin: SqlstormOrigin) -> anyhow::Result<Vec<(usize, String)>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("sqlstorm")
        .join(origin.name());
    let mut entries: Vec<(usize, String)> = Vec::new();
    for entry in
        fs::read_dir(&dir).map_err(|e| anyhow::anyhow!("reading {}: {e}", dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("sql") {
            continue;
        }
        let id: usize = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("non-numeric query file name: {}", path.display()))?;
        entries.push((id, fs::read_to_string(&path)?));
    }
    entries.sort_by_key(|(id, _)| *id);
    Ok(entries)
}
