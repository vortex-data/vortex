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

/// Data-dir subfolder for the native-geometry Vortex files (`points=native`).
pub const NATIVE_DIR: &str = "vortex-native";

/// Data-dir subfolder for the native-geometry GeoParquet files (`points=native`).
pub const PARQUET_NATIVE_DIR: &str = "parquet-native";

/// Queries wired up to run (0-based, `spatialbench.sql` order): Q0 (radius filter), Q1 (zone
/// point-in-polygon), Q2 (point-to-polygon radius, rewritten from `ST_DWithin`), Q5 (zone stats in a
/// Sedona bounding box — its `ST_Intersects(const, z_boundary)` filter pushes into the `zone` scan;
/// the `ST_Within` half stays a DuckDB spatial join), and Q7 (building join). Q1/Q5 need the
/// externally-sourced `zone` table. The file holds the full suite; the rest need tables/functions not
/// wired yet.
const SUPPORTED_QUERIES: &[usize] = &[0, 1, 2, 5, 7];

/// SpatialBench geospatial benchmark (Apache Sedona): a `trip` point table and `building` polygons,
/// queried with spatial filters and joins. See <https://sedona.apache.org/spatialbench/>.
pub struct SpatialBenchBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
    /// `--opt points=native`: store geometry as native `Point`/`Polygon` (not WKB) and read the
    /// native data dirs. The query dialect is chosen per format in [`Self::query_for_format`], not
    /// by this flag.
    pub native_points: bool,
}

impl SpatialBenchBenchmark {
    pub fn new(
        scale_factor: String,
        use_remote_data_dir: Option<String>,
        native_points: bool,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            data_url: resolve_data_url(
                use_remote_data_dir.as_deref(),
                &format!("spatialbench/{scale_factor}"),
            )?,
            scale_factor,
            native_points,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for SpatialBenchBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        // The file is the WKB dialect (`ST_GeomFromWKB(..)`). The dialect is adapted per format in
        // `query_for_format` — not by `points=` — since whether geometry reads back as `GEOMETRY` or
        // `BLOB` depends on the format, not the storage encoding. Statements are `;`-separated,
        // numbered 0-based in file order; only `SUPPORTED_QUERIES` run.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("spatialbench")
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .split_terminator(';')
            .map(str::trim)
            .map(str::to_string)
            .enumerate()
            .filter(|(idx, _)| SUPPORTED_QUERIES.contains(idx))
            .collect())
    }

    /// Only `points=native` Vortex surfaces geometry as `GEOMETRY`, so it drops the
    /// `ST_GeomFromWKB(..)` wrappers. WKB-stored Vortex (and Parquet) read as `BLOB` and keep them.
    fn query_for_format(&self, query: &str, format: Format) -> String {
        match format {
            Format::OnDiskVortex if self.native_points => strip_wkb_wrappers(query),
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

        if self.native_points {
            let parquet_dir = base_data_dir.join(Format::Parquet.name());
            let native_dir = base_data_dir.join(NATIVE_DIR);
            let parquet_native_dir = base_data_dir.join(PARQUET_NATIVE_DIR);
            // Natively encode every table with geometry columns (trip Point, building/zone Polygon).
            // `zone` is sourced externally (`spatialbench-cli`), so only convert it once its parquet
            // is present.
            let mut tables = vec![Table::Trip, Table::Building];
            if zone_parquet_present(&parquet_dir) {
                tables.push(Table::Zone);
            }
            for table in tables {
                datagen::write_native_vortex(table, &parquet_dir, &native_dir).await?;
                datagen::write_native_parquet(table, &parquet_dir, &parquet_native_dir).await?;
            }
        }
        Ok(())
    }

    fn format_path(&self, format: Format, base_url: &Url) -> anyhow::Result<Url> {
        if self.native_points {
            // points=native reads the native-geometry dirs (Vortex / GeoParquet); other formats
            // would feed WKB to the stripped SQL, so bail.
            let dir = match format {
                Format::OnDiskVortex => NATIVE_DIR,
                Format::Parquet => PARQUET_NATIVE_DIR,
                other => anyhow::bail!(
                    "points=native only supports the vortex and parquet formats, got {other}"
                ),
            };
            return Ok(base_url.join(&format!("{dir}/"))?);
        }
        Ok(base_url.join(&format!("{}/", format.name()))?)
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        // Q0 result count by scale factor (index 0), cross-checked against a brute-force WKB decode.
        match self.scale_factor.as_str() {
            "0.1" => Some(vec![6]),
            "1.0" => Some(vec![94]),
            "3.0" => Some(vec![267]),
            _ => None,
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::SpatialBench {
            scale_factor: self.scale_factor.clone(),
            native_points: self.native_points,
        }
    }

    fn dataset_name(&self) -> &str {
        "spatialbench"
    }

    fn dataset_display(&self) -> String {
        format!("spatialbench(sf={})", self.scale_factor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        let mut specs = vec![TableSpec::new("trip", None), TableSpec::new("building", None)];
        // `zone` is externally sourced and optional; register it only when present so Q0/Q7 (which
        // don't need it) don't fail on the missing glob.
        let zone_present = match self.data_url.to_file_path() {
            Ok(base) => zone_parquet_present(&base.join(Format::Parquet.name())),
            Err(()) => true,
        };
        if zone_present {
            specs.push(TableSpec::new("zone", None));
        }
        specs
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
/// upstream `spatialbench-cli`; see the module docs). Native conversion of `zone` is skipped until
/// it is present, so Q0/Q7 runs don't require it.
fn zone_parquet_present(parquet_dir: &std::path::Path) -> bool {
    glob::glob(&parquet_dir.join("zone_*.parquet").to_string_lossy())
        .map(|mut paths| paths.next().is_some())
        .unwrap_or(false)
}

/// Drop each `ST_GeomFromWKB(col)` wrapper down to `col`: native columns are already geometries.
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
