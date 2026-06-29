// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench benchmark implementation

use std::fs;
use std::path::Path;

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

/// Data-dir subfolder for the native-geometry Vortex files (the `vortex-native` lane).
pub const NATIVE_DIR: &str = "vortex-native";

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

    /// Tables to materialize and register: the in-process base tables (`trip`, `building`,
    /// `customer`) plus the externally-sourced `zone` when its parquet is present. Shared by native
    /// data-gen and table registration so both lanes cover the same set.
    fn base_tables(&self) -> Vec<Table> {
        let mut tables = vec![Table::Trip, Table::Building, Table::Customer];
        let zone_present = match self.data_url.to_file_path() {
            Ok(base) => zone_parquet_present(&base.join(Format::Parquet.name())),
            Err(()) => true,
        };
        if zone_present {
            tables.push(Table::Zone);
        }
        tables
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

    /// On the `vortex-native` lane, geometry columns surface as `GEOMETRY`, so drop the
    /// `ST_GeomFromWKB(..)` wrappers and let DuckDB's `spatial` extension evaluate the `ST_*`
    /// predicates directly on the native geometry.
    fn query_for_format(&self, query: &str, format: Format) -> String {
        match format {
            Format::VortexNative => strip_wkb_wrappers(query),
            _ => query.to_string(),
        }
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url.as_str()))?;
        datagen::generate_tables(&self.scale_factor, base_data_dir.clone()).await?;
        Ok(())
    }

    /// The `vortex-native` lane decodes each table's WKB geometry to native GeoArrow once, into the
    /// `vortex-native` dir, so its queries read DuckDB `GEOMETRY` directly. Idempotent.
    async fn prepare_format(&self, format: Format, base_path: &Path) -> anyhow::Result<()> {
        if format == Format::VortexNative {
            let parquet_dir = base_path.join(Format::Parquet.name());
            let native_dir = base_path.join(NATIVE_DIR);
            for table in self.base_tables() {
                datagen::write_native_vortex(table, &parquet_dir, &native_dir).await?;
            }
        }
        Ok(())
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    /// The `vortex-native` lane reads the native-geometry Vortex dir; every other format reads its
    /// own `{format}` subfolder.
    fn format_path(&self, format: Format, base_url: &Url) -> anyhow::Result<Url> {
        let dir = match format {
            Format::VortexNative => NATIVE_DIR,
            other => other.name(),
        };
        Ok(base_url.join(&format!("{dir}/"))?)
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

    /// Both lanes register the same tables (WKB reads `parquet`/`vortex`, native reads
    /// `vortex-native`); `zone` is externally sourced and optional, registered only when present.
    fn table_specs(&self) -> Vec<TableSpec> {
        self.base_tables()
            .iter()
            .map(|table| TableSpec::new(table.name(), None))
            .collect()
    }

    /// DuckDB's view star-expansion drops native `GEOMETRY` columns down to `BLOB`, so `ST_*` fail to
    /// bind. Re-cast every geometry column back to `GEOMETRY` in the view's projection.
    fn view_projection(&self, table_name: &str, format: Format) -> String {
        if format == Format::VortexNative
            && let Some(table) = Table::from_name(table_name)
        {
            let geometry_columns = table.geometry_columns();
            if !geometry_columns.is_empty() {
                let casts = geometry_columns
                    .iter()
                    .map(|column| format!("{name}::GEOMETRY AS {name}", name = column.name))
                    .collect::<Vec<_>>()
                    .join(", ");
                return format!("* REPLACE ({casts})");
            }
        }
        "*".to_string()
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
fn zone_parquet_present(parquet_dir: &Path) -> bool {
    glob::glob(&parquet_dir.join("zone_*.parquet").to_string_lossy())
        .map(|mut paths| paths.next().is_some())
        .unwrap_or(false)
}

/// Strip `ST_GeomFromWKB(<inner>)` → `<inner>` so the native lane reads the already-`GEOMETRY`
/// column directly. Assumes the wrapped expression contains no inner `)` (true for our column refs).
fn strip_wkb_wrappers(sql: &str) -> String {
    const OPEN: &str = "ST_GeomFromWKB(";
    let mut out = String::with_capacity(sql.len());
    let mut rest = sql;
    while let Some(pos) = rest.find(OPEN) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + OPEN.len()..];
        match after.find(')') {
            Some(close) => {
                out.push_str(&after[..close]);
                rest = &after[close + 1..];
            }
            // Unbalanced wrapper: emit it verbatim and stop rewriting.
            None => {
                out.push_str(OPEN);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}
