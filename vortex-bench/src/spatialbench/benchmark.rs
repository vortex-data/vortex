// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench benchmark implementation

use std::fs;

use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;
use crate::TableSpec;
use crate::spatialbench::datagen;
use crate::spatialbench::datagen::Table;
use crate::utils::file::resolve_data_url;
use crate::workspace_root;

/// SpatialBench geospatial benchmark (Apache Sedona): a `trip` point table, `building` polygons, and
/// a `customer` attribute table, queried with spatial filters and joins. `zone` polygons are sourced
/// externally and registered when present. See <https://sedona.apache.org/spatialbench/>.
pub struct SpatialBenchBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl SpatialBenchBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            data_url: resolve_data_url(
                use_remote_data_dir.as_deref(),
                &format!("spatialbench/{scale_factor}"),
            )?,
            scale_factor,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for SpatialBenchBenchmark {
    /// All SpatialBench queries, numbered Q1.. in `spatialbench.sql` file order (1-based, matching
    /// canonical SpatialBench). Geometry is stored as WKB and read back as a DuckDB `BLOB` (via
    /// `ST_GeomFromWKB`), so the `spatial` extension evaluates every `ST_*` predicate — no native
    /// geometry support is needed on the Vortex side.
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        // `;`-separated; a `;` must not appear in a comment, or it would split a statement in two.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("spatialbench")
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .split_terminator(';')
            .map(str::trim)
            .filter(|stmt| !stmt.is_empty())
            .enumerate()
            .map(|(idx, stmt)| (idx + 1, stmt.to_string()))
            .collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url.as_str()))?;
        datagen::generate_tables(&self.scale_factor, base_data_dir).await?;
        Ok(())
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        // Indexed by `query_idx` (1-based), so index 0 is a dummy and Q1's count is at index 1 (TPC-H
        // convention). Only SF1.0 and SF10.0 are validated (like TPC-H); other scale factors return
        // `None`. Each vec covers Q1..Q9 — the queries that finish — and is identical for Parquet and
        // Vortex. Q10..Q12 are heavy spatial joins that time out, so they are left unvalidated (a
        // shorter vec means the runner skips them).
        match self.scale_factor.as_str() {
            "1.0" => Some(vec![0, 94, 1, 22, 258, 316691, 3, 6000000, 369, 37]),
            "10.0" => Some(vec![0, 994, 1, 79, 231, 3144328, 3, 60000000, 9357, 573]),
            _ => None,
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::SpatialBench {
            scale_factor: self.scale_factor.clone(),
        }
    }

    fn dataset_name(&self) -> &str {
        "spatialbench"
    }

    fn dataset_display(&self) -> String {
        format!("spatialbench(sf={})", self.scale_factor)
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        // `zone` is externally sourced and optional; register it only when present so queries that
        // don't need it don't fail on the missing glob.
        let zone_present = match self.data_url.to_file_path() {
            Ok(base) => zone_parquet_present(&base.join(Format::Parquet.name())),
            Err(()) => true,
        };
        Table::ALL
            .into_iter()
            .filter(|table| !matches!(table, Table::Zone) || zone_present)
            .map(|table| TableSpec::new(table.name(), None))
            .collect()
    }

    /// Scope each table to its own `{table}_*.{ext}` files; the default globs every file in the
    /// format dir, conflating the `trip` and `building` schemas.
    fn pattern(&self, table_name: &str, format: Format) -> Option<glob::Pattern> {
        Some(
            format!("{}_*.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }

    /// DuckDB needs the `spatial` extension for `ST_*`; the runner replays it on each (re)open.
    /// First INSTALL needs network.
    fn engine_init_sql(&self, engine: Engine) -> Vec<String> {
        match engine {
            Engine::DuckDB => vec!["INSTALL spatial;".to_string(), "LOAD spatial;".to_string()],
            _ => Vec::new(),
        }
    }
}

/// Whether an externally-sourced `zone_*.parquet` exists under `parquet_dir` (generated by the
/// upstream `spatialbench-cli`; see the module docs).
fn zone_parquet_present(parquet_dir: &std::path::Path) -> bool {
    glob::glob(&parquet_dir.join("zone_*.parquet").to_string_lossy())
        .map(|mut paths| paths.next().is_some())
        .unwrap_or(false)
}
