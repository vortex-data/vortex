// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench benchmark implementation

use std::fs;

use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::spatialbench::datagen;
use crate::spatialbench::datagen::Table;
use crate::utils::file::resolve_data_url;
use crate::workspace_root;

/// Directory under the benchmark data dir holding the native-Point vortex files.
pub const NATIVE_DIR: &str = "vortex-native";

/// Directory under the benchmark data dir holding the native-Point GeoParquet files.
pub const PARQUET_NATIVE_DIR: &str = "parquet-native";

/// SpatialBench geospatial analytics benchmark (Apache Sedona).
///
/// A ride-sharing workload: a `trip` fact table of WKB point locations plus polygon dimension
/// tables, with spatial-predicate, KNN, and join queries. See
/// <https://sedona.apache.org/spatialbench/>.
///
/// Only Q1 and the `trip` table it reads are wired up so far; dimension tables come with later
/// queries.
pub struct SpatialBenchBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
    /// Store geometry as the native Point extension instead of WKB (`--opt points=native`): the
    /// vortex/parquet formats read the native-Point files and queries skip `ST_GeomFromWKB`.
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
        // Queries are in the WKB (canonical SpatialBench) dialect; for `points=native` the
        // `ST_GeomFromWKB(..)` wrappers are stripped. Split on `;`, so no `;` inside a comment.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("spatialbench")
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        let contents = if self.native_points {
            strip_wkb_wrappers(&contents)
        } else {
            contents
        };
        Ok(contents
            .trim()
            .split_terminator(';')
            .map(str::to_string)
            .enumerate()
            .map(|(idx, query)| (idx + 1, query))
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

        datagen::generate_tables(&self.scale_factor, base_data_dir.clone()).await?;

        if self.native_points {
            let parquet_dir = base_data_dir.join(Format::Parquet.name());
            datagen::write_native_vortex(
                Table::Trip,
                &parquet_dir,
                &base_data_dir.join(NATIVE_DIR),
            )
            .await?;
            datagen::write_native_parquet(
                Table::Trip,
                &parquet_dir,
                &base_data_dir.join(PARQUET_NATIVE_DIR),
            )
            .await?;
        }
        Ok(())
    }

    fn format_path(&self, format: Format, base_url: &Url) -> anyhow::Result<Url> {
        if self.native_points {
            // points=native serves the pre-converted native-Point dirs: vortex (Point extension,
            // GeoDistance pushdown) and parquet (GeoParquet geodatafusion reads as geometry).
            // Other formats would feed WKB to native-variant SQL, so fail fast.
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
        // Index 0 is a dummy so Q1's count sits at index 1; counts cross-checked against an
        // independent brute-force WKB decode.
        match self.scale_factor.as_str() {
            "0.1" => Some(vec![0, 6]),
            "1.0" => Some(vec![0, 94]),
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
        vec![TableSpec::new("trip", None)]
    }
}

/// Rewrite a WKB-dialect query for the native-Point encoding by dropping each
/// `ST_GeomFromWKB(col)` wrapper down to `col` -- the native columns are already geometries.
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
