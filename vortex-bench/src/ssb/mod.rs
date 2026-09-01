// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Star Schema Benchmark (SSB).
//!
//! A denormalized redesign of TPC-H into a classic star schema: one wide `lineorder` fact table
//! joined against four dimensions (`customer`, `supplier`, `part`, `dwdate`). The 13 queries are
//! organized into four "flights" of progressively more selective dimension filters, which makes
//! the suite a direct test of filter pushdown, zone-map pruning, and dimension-join throughput —
//! the axes on which a columnar format is supposed to win.
//!
//! Data generation lives in [`datagen`], over the native generator in `ssbgen`.

use std::fs;
use std::path::Path;

use glob::Pattern;
use url::Url;
use vortex::error::VortexExpect;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::datasets::SSB_TABLES;
use crate::utils::file::resolve_data_url;

pub mod datagen;
pub(crate) mod ssbgen;

/// The 13 SSB queries, stored as `q1.sql` ... `q13.sql`. The framework keys queries on a plain
/// index, so the flight-and-query numbering from the paper (Q1.1 ... Q4.3) maps onto 1 ... 13 in
/// order; each file names its SSB query in a leading comment, and `sql/ssb/README.md` carries the
/// full table.
pub fn ssb_queries() -> impl Iterator<Item = (usize, String)> {
    (1..=13).map(|q| (q, ssb_query(q)))
}

fn ssb_query(query_idx: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("sql")
        .join("ssb")
        .join(format!("q{query_idx}"))
        .with_extension("sql");
    fs::read_to_string(path).vortex_expect("cannot load ssb query from file")
}

/// Benchmark over the [Star Schema Benchmark][ssb].
///
/// [ssb]: https://www.cs.umb.edu/~poneil/StarSchemaB.PDF
pub struct SsbBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl SsbBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            data_url: resolve_data_url(
                use_remote_data_dir.as_deref(),
                &format!("ssb/{scale_factor}"),
            )?,
            scale_factor,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for SsbBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/ssb/README.md"
    }

    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(ssb_queries().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }
        let base_dir = self.data_url.to_file_path().map_err(|()| {
            anyhow::anyhow!(
                "Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"
            )
        })?;
        datagen::generate_tables(&self.scale_factor, &base_dir)
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        // Indexed by `query_idx` (1-based), so index 0 is a dummy and Q1's count is at index 1
        // (TPC-H convention). Only the scale factors CI runs are validated; anything else
        // returns `None`. Measured with DuckDB over the generated Parquet, and consistent with
        // the group cardinalities the SSB schema implies — e.g. Q4.3 at SF 10 is 2 years x 10
        // US cities x 40 `MFGR#14` brands = 800.
        match self.scale_factor.as_str() {
            "1.0" => Some(vec![0, 1, 1, 1, 280, 56, 7, 150, 600, 24, 3, 35, 100, 725]),
            "10.0" => Some(vec![0, 1, 1, 1, 280, 56, 7, 150, 600, 24, 4, 35, 100, 800]),
            _ => None,
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Ssb {
            scale_factor: self.scale_factor.clone(),
        }
    }

    fn dataset_name(&self) -> &str {
        "ssb"
    }

    fn dataset_display(&self) -> String {
        format!("ssb(sf={})", self.scale_factor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        SSB_TABLES
            .iter()
            .map(|name| TableSpec::new(name, None))
            .collect()
    }

    /// Scope each table to its own file; the default globs every file in the format dir, which
    /// would conflate the five schemas.
    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            format!("{}.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
