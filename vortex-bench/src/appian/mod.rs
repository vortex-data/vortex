// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Appian benchmark.
//!
//! Mirrors the queries from DuckDB's in-tree `benchmark/appian_benchmarks` suite. Upstream
//! ships the data as a single `.duckdb` blob (~593 MB); we download it once and shell out
//! to the `duckdb` CLI to project each table into Parquet, lowercasing column names along
//! the way. `data-gen` then handles every other format from those Parquet files.
//!
//! ## Identifier case
//!
//! The upstream `.duckdb` blob preserves camelCase column names (`orderItem_quantity`,
//! `address_customerId`, ...) and capitalized table names (`CustomerView`). The Appian
//! queries reference those identifiers unquoted, which would break under DataFusion's
//! default `enable_ident_normalization=true` (parser lowercases identifier references
//! while the Parquet schema and registered table names preserve case → field-not-found).
//!
//! The conversion below lowercases every column at COPY time, and the table names in
//! `TABLES` are already lowercase. Both engines then resolve the verbatim camelCase
//! queries the same way: DataFusion lowercases the query identifiers and matches them
//! against the lowercased Parquet schema, while DuckDB's case-insensitive unquoted
//! identifier resolution makes the original case irrelevant.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::bail;
use glob::Pattern;
use tracing::info;
use url::Url;
use vortex::error::VortexExpect;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::datasets::data_downloads::download_data;
use crate::utils::file::resolve_data_url;

/// Upstream `.duckdb` blob; pinned to the URL hard-coded into DuckDB's
/// `benchmark/appian_benchmarks/appian.benchmark.in`.
const UPSTREAM_BLOB_URL: &str = "https://blobs.duckdb.org/data/appian_benchmark_data.duckdb";

/// Table names from DuckDB's `appian.benchmark.in` template in upstream case. Ordering
/// must match [`TABLES`] so each upstream source maps to its lowercased Parquet output.
const UPSTREAM_TABLES: &[&str] = &[
    "AddressView",
    "CategoryView",
    "CreditCardView",
    "CustomerView",
    "OrderItemNovelty_Update",
    "OrderItemView",
    "OrderView",
    "ProductView",
    "TaxRecordView",
];

/// Lowercased table names registered with the query engines. Matches the output Parquet
/// file names produced by [`AppianBenchmark::generate_base_data`].
const TABLES: &[&str] = &[
    "addressview",
    "categoryview",
    "creditcardview",
    "customerview",
    "orderitemnovelty_update",
    "orderitemview",
    "orderview",
    "productview",
    "taxrecordview",
];

/// Eight join-heavy queries from `duckdb/duckdb:benchmark/appian_benchmarks/queries/`,
/// stored byte-identically under `vortex-bench/appian/q{1..8}.sql` (sibling of the TPC-H
/// `tpch/q*.sql` layout). Upstream refreshes are a pure copy into that directory.
pub fn appian_queries() -> impl Iterator<Item = (usize, String)> {
    (1..=8).map(|q| (q, appian_query(q)))
}

fn appian_query(query_idx: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("appian")
        .join(format!("q{query_idx}"))
        .with_extension("sql");
    fs::read_to_string(path).vortex_expect("cannot load appian query from file")
}

/// Benchmark over the [Appian benchmark suite from DuckDB][upstream].
///
/// [upstream]: https://github.com/duckdb/duckdb/tree/main/benchmark/appian_benchmarks
pub struct AppianBenchmark {
    data_url: Url,
}

impl AppianBenchmark {
    pub fn new(data_url: Url) -> Self {
        Self { data_url }
    }

    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = resolve_data_url(use_remote_data_dir.as_deref(), "appian")?;
        Ok(Self { data_url })
    }

    fn base_dir(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!(
                "Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"
            ))
    }

    fn parquet_dir(&self) -> anyhow::Result<PathBuf> {
        Ok(self.base_dir()?.join(Format::Parquet.name()))
    }
}

#[async_trait::async_trait]
impl Benchmark for AppianBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(appian_queries().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let parquet_dir = self.parquet_dir()?;
        fs::create_dir_all(&parquet_dir)?;

        // Idempotency: if every target Parquet is already in place, do nothing.
        if TABLES
            .iter()
            .all(|t| parquet_dir.join(format!("{t}.parquet")).exists())
        {
            info!(
                "appian: {} Parquet shards already present in {}",
                TABLES.len(),
                parquet_dir.display(),
            );
            return Ok(());
        }

        // Download the upstream `.duckdb` blob into the dataset cache directory.
        let blob_path = self.base_dir()?.join("appian_benchmark_data.duckdb");
        let blob = download_data(blob_path, UPSTREAM_BLOB_URL).await?;

        // DuckDB SQL can't use a query result as a projection list, so build per-table
        // lowercased projections in Rust, then run all nine `COPY`s in a single subprocess.
        let projections = discover_projections(&blob)?;
        let mut script = format!("ATTACH '{}' AS src (READ_ONLY);\n", blob.display());
        for (i, &upstream) in UPSTREAM_TABLES.iter().enumerate() {
            let projection = projections
                .iter()
                .find(|(t, _)| t == upstream)
                .map(|(_, p)| p.as_str())
                .with_context(|| format!("no columns reported for upstream table {upstream}"))?;
            let out_path = parquet_dir.join(format!("{}.parquet", TABLES[i]));
            script.push_str(&format!(
                "COPY (SELECT {projection} FROM src.\"{upstream}\") TO '{}' (FORMAT PARQUET);\n",
                out_path.display(),
            ));
        }

        let output = Command::new("duckdb").arg("-c").arg(&script).output()?;
        if !output.status.success() {
            bail!(
                "duckdb appian COPY failed: stdout={:?} stderr={:?}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        info!(
            "appian base data generated in {} ({} Parquet shards)",
            parquet_dir.display(),
            TABLES.len(),
        );
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Appian
    }

    fn dataset_name(&self) -> &str {
        "appian"
    }

    fn dataset_display(&self) -> String {
        "appian".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        TABLES
            .iter()
            .map(|name| TableSpec::new(name, None))
            .collect()
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            format!("{}.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}

/// Run a single `duckdb` invocation that returns, for each upstream Appian table, a
/// projection string of the form `"OrigName" AS "origname", ...` so the `COPY` statements
/// below can lowercase every column name without enumerating them by hand.
fn discover_projections(blob: &Path) -> anyhow::Result<Vec<(String, String)>> {
    // `chr(31)` (unit separator) keeps `table_name` and the projection list distinct in
    // the single-column `-list` output without colliding with `|` (list separator) or
    // `,` (projection delimiter).
    let sql = format!(
        "ATTACH '{}' AS src (READ_ONLY); \
         SELECT table_name || chr(31) || \
                string_agg('\"' || column_name || '\" AS \"' || lower(column_name) || '\"', ', ' ORDER BY column_index) \
         FROM duckdb_columns() \
         WHERE database_name = 'src' \
         GROUP BY table_name;",
        blob.display(),
    );
    let output = Command::new("duckdb")
        .arg("-noheader")
        .arg("-list")
        .arg("-c")
        .arg(&sql)
        .output()?;
    if !output.status.success() {
        bail!(
            "duckdb column discovery failed: stdout={:?} stderr={:?}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    let stdout = String::from_utf8(output.stdout)?;
    Ok(stdout
        .lines()
        .filter_map(|line| {
            line.split_once('\x1f')
                .map(|(t, p)| (t.to_owned(), p.to_owned()))
        })
        .collect())
}
