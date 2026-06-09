// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Verification of a migrated v3 DuckDB against another substrate.
//!
//! Two independent checks live here:
//!
//! 1. [`run`] is a structural diff between a migrated v3 DuckDB and the live v2
//!    `/api/metadata` endpoint. It compares group / chart structure only; values
//!    are not compared because v2 converts ns -> ms and bytes -> MiB on read
//!    while v3 stores raw and the chart query divides. Group / chart structural
//!    equivalence is enough to spot classifier regressions before cutover.
//!
//! 2. [`run_postgres_value_verify`] is the PR-3.2 PRIMARY v4-correctness gate: a
//!    per-`measurement_id` value-column comparison between a DuckDB source and a
//!    Postgres target. `measurement_id` hashes only `commit_sha` and the dim
//!    tuple, so a primary-key / row-count match pins zero bytes of any value or
//!    env column; this check compares every non-hashed column per row instead and
//!    fails on any presence diff or value mismatch.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use arrow_array::Array;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use duckdb::Connection;
use serde::Deserialize;

use crate::classifier::QUERY_SUITES;
use crate::postgres::ColKind;
use crate::postgres::column_kind;
use crate::postgres::connect_postgres;

/// Result of one `verify` run.
#[derive(Debug, Default)]
pub struct VerifyReport {
    /// Group display names present in both v2 and v3.
    pub matched_groups: Vec<String>,
    /// Group display names that exist in v3 but not v2.
    pub only_in_v3: Vec<String>,
    /// Group display names that exist in v2 but not v3 — these gate the CLI's
    /// non-zero exit.
    pub only_in_v2: Vec<String>,
    /// Per-group chart-count diffs for groups present on both sides.
    pub chart_diffs: Vec<ChartDiff>,
}

/// One group's chart-count divergence between v2 and v3, captured when the
/// group is structurally present on both sides but the counts differ.
#[derive(Debug, Clone)]
pub struct ChartDiff {
    /// Group display name.
    pub group: String,
    /// Number of charts v2 reported for this group.
    pub v2_count: usize,
    /// Number of charts the migrated v3 DuckDB has for this group.
    pub v3_count: usize,
}

impl VerifyReport {
    /// True if every v2 group is represented in v3. The CLI's exit
    /// code reflects this.
    pub fn v2_groups_covered(&self) -> bool {
        self.only_in_v2.is_empty()
    }
}

impl std::fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Groups in both v2 and v3:")?;
        for g in &self.matched_groups {
            writeln!(f, "  + {g}")?;
        }
        if !self.only_in_v2.is_empty() {
            writeln!(f, "Groups only in v2 (regression candidates):")?;
            for g in &self.only_in_v2 {
                writeln!(f, "  - {g}")?;
            }
        }
        if !self.only_in_v3.is_empty() {
            writeln!(f, "Groups only in v3:")?;
            for g in &self.only_in_v3 {
                writeln!(f, "  + {g}")?;
            }
        }
        if !self.chart_diffs.is_empty() {
            writeln!(f, "Chart count diffs:")?;
            for d in &self.chart_diffs {
                writeln!(
                    f,
                    "  {} : v2={} v3={} (delta={})",
                    d.group,
                    d.v2_count,
                    d.v3_count,
                    d.v3_count as i64 - d.v2_count as i64,
                )?;
            }
        }
        Ok(())
    }
}

/// v2's `/api/metadata` reply — only the fields we need.
#[derive(Debug, Deserialize)]
struct V2Metadata {
    groups: BTreeMap<String, V2GroupMeta>,
}

#[derive(Debug, Deserialize)]
struct V2GroupMeta {
    #[serde(default)]
    charts: Vec<V2ChartMeta>,
}

#[derive(Debug, Deserialize)]
struct V2ChartMeta {
    #[serde(default)]
    name: String,
}

/// Open the migrated DuckDB at `duckdb_path`, fetch `<v2_server>/api/metadata`,
/// and produce a structural diff.
pub fn run(v2_server: &str, duckdb_path: &Path) -> Result<VerifyReport> {
    let v3 = collect_v3_groups(duckdb_path)?;
    let v2 = fetch_v2_metadata(v2_server)?;
    Ok(diff(&v2, &v3))
}

fn collect_v3_groups(duckdb_path: &Path) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let conn = Connection::open(duckdb_path)
        .with_context(|| format!("opening DuckDB at {}", duckdb_path.display()))?;
    let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    // query_measurements: chart per (dataset, query_idx); group per
    // (dataset, dataset_variant, scale_factor, storage). We want v2
    // group display names so the verifier can compare apples to
    // apples, so we re-format them here using the same suite table.
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant, scale_factor, storage, query_idx
          FROM query_measurements
         GROUP BY dataset, dataset_variant, scale_factor, storage, query_idx
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i32>(4)?,
        ))
    })?;
    for row in rows {
        let (dataset, _variant, sf, storage, query_idx) = row?;
        let group_name = display_query_group(&dataset, sf.as_deref(), &storage);
        let chart_name = chart_name_query(&dataset, query_idx);
        groups
            .entry(group_name)
            .or_default()
            .insert(normalize_chart(&chart_name));
    }

    // compression_times: group "Compression", charts per dataset.
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, format, op
          FROM compression_times
         GROUP BY dataset, format, op
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    for row in rows {
        let (dataset, format, op) = row?;
        let chart = chart_name_compression_time(&format, &op, &dataset);
        groups
            .entry("Compression".to_string())
            .or_default()
            .insert(normalize_chart(&chart));
    }

    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, format
          FROM compression_sizes
         GROUP BY dataset, format
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (_dataset, format) = row?;
        let chart = chart_name_compression_size(&format);
        groups
            .entry("Compression Size".to_string())
            .or_default()
            .insert(normalize_chart(&chart));
    }

    let mut stmt = conn.prepare(
        r#"
        SELECT DISTINCT dataset
          FROM random_access_times
        "#,
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let dataset = row?;
        groups
            .entry("Random Access".to_string())
            .or_default()
            .insert(normalize_chart(&dataset));
    }

    Ok(groups)
}

fn fetch_v2_metadata(server: &str) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let url = format!("{}/api/metadata", server.trim_end_matches('/'));
    let body = reqwest::blocking::get(&url)
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("non-2xx from {url}"))?
        .json::<V2Metadata>()
        .with_context(|| format!("parsing {url} as v2 /api/metadata"))?;
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, group) in body.groups {
        let charts = group
            .charts
            .into_iter()
            .map(|c| normalize_chart(&c.name))
            .collect();
        out.insert(name, charts);
    }
    Ok(out)
}

fn diff(
    v2: &BTreeMap<String, BTreeSet<String>>,
    v3: &BTreeMap<String, BTreeSet<String>>,
) -> VerifyReport {
    let mut report = VerifyReport::default();
    let v2_keys: BTreeSet<&String> = v2.keys().collect();
    let v3_keys: BTreeSet<&String> = v3.keys().collect();
    for g in v2_keys.intersection(&v3_keys) {
        report.matched_groups.push((**g).clone());
        let v2_charts = &v2[*g];
        let v3_charts = &v3[*g];
        if v2_charts.len() != v3_charts.len() {
            report.chart_diffs.push(ChartDiff {
                group: (**g).clone(),
                v2_count: v2_charts.len(),
                v3_count: v3_charts.len(),
            });
        }
    }
    for g in v3_keys.difference(&v2_keys) {
        report.only_in_v3.push((**g).clone());
    }
    for g in v2_keys.difference(&v3_keys) {
        report.only_in_v2.push((**g).clone());
    }
    report.matched_groups.sort();
    report.only_in_v3.sort();
    report.only_in_v2.sort();
    report
}

fn display_query_group(dataset: &str, scale_factor: Option<&str>, storage: &str) -> String {
    let suite = QUERY_SUITES
        .iter()
        .find(|s| s.prefix.eq_ignore_ascii_case(dataset))
        .copied();
    match suite {
        Some(suite) if suite.fan_out => {
            let storage_disp = match storage {
                "s3" | "S3" => "S3",
                _ => "NVMe",
            };
            let sf = scale_factor.unwrap_or("1");
            format!("{} ({}) (SF={})", suite.display_name, storage_disp, sf)
        }
        Some(suite) => suite.display_name.to_string(),
        None => format!("{dataset} ({storage})"),
    }
}

fn chart_name_query(dataset: &str, query_idx: i32) -> String {
    let suite = QUERY_SUITES
        .iter()
        .find(|s| s.prefix.eq_ignore_ascii_case(dataset))
        .copied();
    match suite {
        Some(suite) => format!("{} Q{}", suite.query_prefix, query_idx),
        None => format!("{} Q{}", dataset.to_uppercase(), query_idx),
    }
}

fn chart_name_compression_time(format: &str, op: &str, _dataset: &str) -> String {
    // Re-derive the v2 chart name (the metric, not the dataset) so we
    // can compare. v2's chart axis is the metric; series is the
    // dataset. v3 inverts that. For structural comparison, we project
    // back to v2's per-chart key.
    match (format, op) {
        ("vortex-file-compressed", "encode") => "COMPRESS TIME".into(),
        ("vortex-file-compressed", "decode") => "DECOMPRESS TIME".into(),
        ("parquet", "encode") => "PARQUET RS ZSTD COMPRESS TIME".into(),
        ("parquet", "decode") => "PARQUET RS ZSTD DECOMPRESS TIME".into(),
        ("lance", "encode") => "LANCE COMPRESS TIME".into(),
        ("lance", "decode") => "LANCE DECOMPRESS TIME".into(),
        _ => format!("{} {} TIME", format.to_uppercase(), op.to_uppercase()),
    }
}

fn chart_name_compression_size(format: &str) -> String {
    match format {
        "vortex-file-compressed" => "VORTEX SIZE".into(),
        "parquet" => "PARQUET SIZE".into(),
        "lance" => "LANCE SIZE".into(),
        _ => format!("{} SIZE", format.to_uppercase()),
    }
}

/// Strip casing and `_-` differences between v2 and v3 chart names.
/// v2 displays uppercase; v3 stores raw values. Comparing in this
/// canonical form is enough for structural verification.
fn normalize_chart(s: &str) -> String {
    s.trim()
        .to_uppercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// DuckDB -> Postgres per-`measurement_id` value verification (PR-3.2).
//
// This is the PRIMARY v4-correctness gate. The verifier joins the DuckDB source
// against the Postgres target on `measurement_id` (1:1; the accumulators dedup by
// id) and `commits` on `commit_sha`, then compares every non-hashed value column
// per row. Any presence diff or value mismatch fails the gate. The comparison is
// full, not sampled.
// ---------------------------------------------------------------------------

/// One value-column mismatch between the DuckDB source and the Postgres target
/// for a single keyed row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueMismatch {
    /// Table the mismatch was found in.
    pub table: &'static str,
    /// Join key (a `measurement_id` for the fact tables, a `commit_sha` for
    /// `commits`), rendered for display.
    pub key: String,
    /// The value column that differs.
    pub column: &'static str,
    /// The DuckDB-source value, rendered.
    pub duckdb_value: String,
    /// The Postgres-target value, rendered.
    pub pg_value: String,
}

/// Result of a `verify --postgres-target` run: per-table presence diffs plus
/// per-row value-column mismatches. Distinct from the v2-vs-v3 structural
/// [`VerifyReport`] above, which compares group / chart shape rather than stored
/// values.
#[derive(Debug, Default)]
pub struct PgVerifyReport {
    /// `(table, key)` present in the DuckDB source but absent from Postgres.
    pub only_in_duckdb: Vec<(&'static str, String)>,
    /// `(table, key)` present in Postgres but absent from the DuckDB source.
    pub only_in_postgres: Vec<(&'static str, String)>,
    /// Every per-row value-column mismatch found.
    pub value_mismatches: Vec<ValueMismatch>,
}

impl PgVerifyReport {
    /// True when the source and target match exactly: no presence diff and no
    /// value mismatch. The CLI exit code reflects this.
    pub fn is_clean(&self) -> bool {
        self.only_in_duckdb.is_empty()
            && self.only_in_postgres.is_empty()
            && self.value_mismatches.is_empty()
    }
}

impl std::fmt::Display for PgVerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_clean() {
            return writeln!(
                f,
                "value verify: source and target match (0 presence diffs, 0 value mismatches)"
            );
        }
        if !self.only_in_duckdb.is_empty() {
            writeln!(
                f,
                "Keys only in DuckDB source ({}):",
                self.only_in_duckdb.len()
            )?;
            for (table, key) in &self.only_in_duckdb {
                writeln!(f, "  - {table}: {key}")?;
            }
        }
        if !self.only_in_postgres.is_empty() {
            writeln!(
                f,
                "Keys only in Postgres target ({}):",
                self.only_in_postgres.len()
            )?;
            for (table, key) in &self.only_in_postgres {
                writeln!(f, "  + {table}: {key}")?;
            }
        }
        if !self.value_mismatches.is_empty() {
            writeln!(f, "Value mismatches ({}):", self.value_mismatches.len())?;
            for m in &self.value_mismatches {
                writeln!(
                    f,
                    "  {} [{}] {}: duckdb={} pg={}",
                    m.table, m.key, m.column, m.duckdb_value, m.pg_value
                )?;
            }
        }
        Ok(())
    }
}

/// A single comparable cell value, normalized so the DuckDB and Postgres reads of
/// the same logical value compare equal regardless of source engine.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CellValue {
    /// SQL NULL.
    Null,
    /// An `INTEGER` / `BIGINT` value, or a `TIMESTAMPTZ` rendered as epoch
    /// microseconds. `i32` columns are widened to `i64`.
    Int(i64),
    /// A `TEXT` value, compared byte for byte.
    Text(String),
    /// A `BIGINT[]` value, compared element-wise (order-sensitive).
    Array(Vec<i64>),
}

impl std::fmt::Display for CellValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellValue::Null => write!(f, "NULL"),
            CellValue::Int(n) => write!(f, "{n}"),
            // Quote text so an empty string or trailing whitespace stays visible
            // in a mismatch line.
            CellValue::Text(s) => write!(f, "{s:?}"),
            CellValue::Array(v) => write!(f, "{v:?}"),
        }
    }
}

/// A row's join key. Ordered + comparable so rows from each side line up in a
/// `BTreeMap`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum RowKey {
    /// A fact table's `measurement_id`.
    Id(i64),
    /// A `commits.commit_sha`.
    Sha(String),
}

impl std::fmt::Display for RowKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowKey::Id(id) => write!(f, "{id}"),
            RowKey::Sha(sha) => write!(f, "{sha}"),
        }
    }
}

/// Which column a table joins on for the value comparison.
#[derive(Debug, Clone, Copy)]
enum KeyKind {
    /// `measurement_id` (`BIGINT`); the five fact tables.
    MeasurementId,
    /// `commit_sha` (`TEXT`); the `commits` table.
    CommitSha,
}

impl KeyKind {
    fn column(self) -> &'static str {
        match self {
            KeyKind::MeasurementId => "measurement_id",
            KeyKind::CommitSha => "commit_sha",
        }
    }
}

/// How a value column is read and compared.
#[derive(Debug, Clone, Copy)]
enum Compare {
    /// Read + compare using the loader's authoritative column kind
    /// ([`column_kind`]).
    ByKind,
    /// A `TIMESTAMPTZ` compared as epoch microseconds. Epoch is engine and
    /// timezone independent, so it sidesteps the DuckDB-vs-Postgres divergence in
    /// timestamp text rendering. Only `commits.timestamp` uses this.
    EpochMicros,
}

/// One value column to compare.
#[derive(Debug, Clone, Copy)]
struct ValueCol {
    name: &'static str,
    compare: Compare,
}

const fn vc(name: &'static str) -> ValueCol {
    ValueCol {
        name,
        compare: Compare::ByKind,
    }
}

const fn ts(name: &'static str) -> ValueCol {
    ValueCol {
        name,
        compare: Compare::EpochMicros,
    }
}

/// One table's value-verify plan: the join key + the value (non-key, non-dim)
/// columns to compare. Dim columns are intentionally absent; they are encoded
/// into `measurement_id`, so a dim divergence surfaces as a presence diff, not a
/// value mismatch.
struct ValueSpec {
    table: &'static str,
    key: KeyKind,
    value_columns: &'static [ValueCol],
}

/// The six tables' value-verify plans. The value-column lists are the
/// "non-hashed columns" enumerated in the plan's PR-3.2 row.
const VALUE_SPECS: &[ValueSpec] = &[
    ValueSpec {
        table: "commits",
        key: KeyKind::CommitSha,
        value_columns: &[
            ts("timestamp"),
            vc("message"),
            vc("author_name"),
            vc("author_email"),
            vc("committer_name"),
            vc("committer_email"),
            vc("tree_sha"),
            vc("url"),
        ],
    },
    ValueSpec {
        table: "query_measurements",
        key: KeyKind::MeasurementId,
        value_columns: &[
            vc("value_ns"),
            vc("all_runtimes_ns"),
            vc("peak_physical"),
            vc("peak_virtual"),
            vc("physical_delta"),
            vc("virtual_delta"),
            vc("env_triple"),
        ],
    },
    ValueSpec {
        table: "compression_times",
        key: KeyKind::MeasurementId,
        value_columns: &[vc("value_ns"), vc("all_runtimes_ns"), vc("env_triple")],
    },
    ValueSpec {
        table: "compression_sizes",
        key: KeyKind::MeasurementId,
        value_columns: &[vc("value_bytes")],
    },
    ValueSpec {
        table: "random_access_times",
        key: KeyKind::MeasurementId,
        value_columns: &[vc("value_ns"), vc("all_runtimes_ns"), vc("env_triple")],
    },
    ValueSpec {
        table: "vector_search_runs",
        key: KeyKind::MeasurementId,
        value_columns: &[
            vc("value_ns"),
            vc("all_runtimes_ns"),
            vc("matches"),
            vc("rows_scanned"),
            vc("bytes_scanned"),
            vc("iterations"),
            vc("env_triple"),
        ],
    },
];

/// How one value column is read from a result row, derived from the loader's
/// column kind (or fixed to `Int64` for an epoch-microseconds timestamp).
#[derive(Debug, Clone, Copy)]
enum ReadKind {
    /// `INTEGER`, widened to `i64`.
    Int32,
    /// `BIGINT`, or epoch microseconds.
    Int64,
    /// `TEXT`.
    Text,
    /// `BIGINT[]`.
    IntArray,
}

/// Resolve, once per table, how each of `spec`'s value columns is read. Errors if
/// a value column is not a known loader column or is an unsupported f64 (a float
/// column must be a hashed dim, never a compared value).
fn read_kinds(spec: &ValueSpec) -> Result<Vec<ReadKind>> {
    spec.value_columns
        .iter()
        .map(|vcol| match vcol.compare {
            Compare::EpochMicros => Ok(ReadKind::Int64),
            Compare::ByKind => match column_kind(spec.table, vcol.name) {
                Some(ColKind::Int) => Ok(ReadKind::Int32),
                Some(ColKind::BigInt) => Ok(ReadKind::Int64),
                Some(ColKind::Text) => Ok(ReadKind::Text),
                Some(ColKind::BigIntArray) => Ok(ReadKind::IntArray),
                Some(ColKind::Double) => bail!(
                    "verify: {}.{} is an f64 column; the value verifier does not \
                     compare floats (a float column should be a hashed dim, not a value)",
                    spec.table,
                    vcol.name
                ),
                None => bail!(
                    "verify: {}.{} is not a known loader column",
                    spec.table,
                    vcol.name
                ),
            },
        })
        .collect()
}

/// Value-verify the DuckDB snapshot at `duckdb_path` against the Postgres target
/// at `dsn`. TLS is selected by `ca_cert` exactly as in [`crate::postgres::load`].
pub fn run_postgres_value_verify(
    duckdb_path: &Path,
    dsn: &str,
    ca_cert: Option<&Path>,
) -> Result<PgVerifyReport> {
    let duck_config = duckdb::Config::default()
        .access_mode(duckdb::AccessMode::ReadOnly)
        .context("configuring read-only DuckDB access")?;
    let duck = duckdb::Connection::open_with_flags(duckdb_path, duck_config)
        .with_context(|| format!("opening source DuckDB at {}", duckdb_path.display()))?;
    // No `SET TimeZone='UTC'` is needed here (unlike the loader): epoch microseconds
    // are timezone-independent, and this read renders no column as timestamp text.
    let mut pg = connect_postgres(dsn, ca_cert)?;

    let mut report = PgVerifyReport::default();
    for spec in VALUE_SPECS {
        let kinds = read_kinds(spec)?;
        let duck_rows = read_duck_table(&duck, spec, &kinds)?;
        let pg_rows = read_pg_table(&mut pg, spec, &kinds)?;
        compare_table(spec, &duck_rows, &pg_rows, &mut report);
    }
    Ok(report)
}

/// Build a `SELECT key, value-cols... FROM table`. `epoch_expr` renders a
/// `Compare::EpochMicros` column into engine-specific epoch-microsecond SQL; every
/// other column is read by its bare name. The key is always column 0.
fn select_sql(spec: &ValueSpec, epoch_expr: impl Fn(&str) -> String) -> String {
    let mut cols = vec![spec.key.column().to_string()];
    for vcol in spec.value_columns {
        cols.push(match vcol.compare {
            Compare::EpochMicros => format!("{} AS {}", epoch_expr(vcol.name), vcol.name),
            Compare::ByKind => vcol.name.to_string(),
        });
    }
    format!("SELECT {} FROM {}", cols.join(", "), spec.table)
}

/// DuckDB read SQL. `epoch_us(ts)` returns `BIGINT` microseconds since the Unix
/// epoch, independent of session timezone.
fn duck_select_sql(spec: &ValueSpec) -> String {
    select_sql(spec, |c| format!("epoch_us({c})"))
}

/// Postgres read SQL. Postgres has no `epoch_us`; `extract(epoch from ts)` is the
/// portable epoch extractor. On PostgreSQL 14+ (the RDS target is 16) it returns
/// `numeric`, so `* 1000000` is exact integer-valued `numeric` arithmetic and
/// `::bigint` is lossless for any microsecond-precision timestamp. The result
/// matches DuckDB's integer `epoch_us` to the microsecond. (On pre-14 Postgres
/// `extract(epoch ...)` is `double precision`, still exact for the whole-second
/// git commit timestamps this loads: their integer-valued `f64` epoch times an
/// exact `1e6` stays an integer below 2^53.)
fn pg_select_sql(spec: &ValueSpec) -> String {
    select_sql(spec, |c| {
        format!("(extract(epoch from {c}) * 1000000)::bigint")
    })
}

/// Read `spec`'s key + value columns from DuckDB into a keyed map of value cells
/// (aligned to `spec.value_columns`).
fn read_duck_table(
    duck: &duckdb::Connection,
    spec: &ValueSpec,
    kinds: &[ReadKind],
) -> Result<BTreeMap<RowKey, Vec<CellValue>>> {
    let sql = duck_select_sql(spec);
    let mut stmt = duck
        .prepare(&sql)
        .with_context(|| format!("preparing `{sql}`"))?;
    let batches: Vec<RecordBatch> = stmt
        .query_arrow([])
        .with_context(|| format!("reading {} from DuckDB", spec.table))?
        .collect();

    let mut rows = BTreeMap::new();
    let expected = 1 + spec.value_columns.len();
    for batch in &batches {
        if batch.num_columns() != expected {
            bail!(
                "table {}: DuckDB returned {} columns, expected {}",
                spec.table,
                batch.num_columns(),
                expected
            );
        }
        for row in 0..batch.num_rows() {
            let key = arrow_key(spec.key, batch.column(0).as_ref(), row)
                .with_context(|| format!("reading {} key", spec.table))?;
            let mut cells = Vec::with_capacity(spec.value_columns.len());
            for (idx, kind) in kinds.iter().enumerate() {
                let array = batch.column(idx + 1).as_ref();
                let cell = arrow_cell(array, row, *kind).with_context(|| {
                    format!("reading {}.{}", spec.table, spec.value_columns[idx].name)
                })?;
                cells.push(cell);
            }
            if rows.insert(key.clone(), cells).is_some() {
                bail!(
                    "table {}: duplicate key {} in the DuckDB source (expected 1:1 by id)",
                    spec.table,
                    key
                );
            }
        }
    }
    Ok(rows)
}

/// Read `spec`'s key + value columns from Postgres into a keyed map of value cells.
fn read_pg_table(
    pg: &mut postgres::Client,
    spec: &ValueSpec,
    kinds: &[ReadKind],
) -> Result<BTreeMap<RowKey, Vec<CellValue>>> {
    let sql = pg_select_sql(spec);
    let pg_rows = pg
        .query(sql.as_str(), &[])
        .with_context(|| format!("reading {} from Postgres", spec.table))?;

    let mut rows = BTreeMap::new();
    for row in &pg_rows {
        let key = pg_key(spec.key, row).with_context(|| format!("reading {} key", spec.table))?;
        let mut cells = Vec::with_capacity(spec.value_columns.len());
        for (idx, kind) in kinds.iter().enumerate() {
            let cell = pg_cell(row, idx + 1, *kind).with_context(|| {
                format!("reading {}.{}", spec.table, spec.value_columns[idx].name)
            })?;
            cells.push(cell);
        }
        // The primary key enforces uniqueness, so a duplicate is impossible; a
        // plain insert is correct.
        rows.insert(key, cells);
    }
    Ok(rows)
}

/// Compare one table's source and target keyed maps, appending any presence diffs
/// + value mismatches to `report`.
fn compare_table(
    spec: &ValueSpec,
    duck_rows: &BTreeMap<RowKey, Vec<CellValue>>,
    pg_rows: &BTreeMap<RowKey, Vec<CellValue>>,
    report: &mut PgVerifyReport,
) {
    for (key, duck_cells) in duck_rows {
        match pg_rows.get(key) {
            None => report.only_in_duckdb.push((spec.table, key.to_string())),
            Some(pg_cells) => {
                for (idx, vcol) in spec.value_columns.iter().enumerate() {
                    if duck_cells[idx] != pg_cells[idx] {
                        report.value_mismatches.push(ValueMismatch {
                            table: spec.table,
                            key: key.to_string(),
                            column: vcol.name,
                            duckdb_value: duck_cells[idx].to_string(),
                            pg_value: pg_cells[idx].to_string(),
                        });
                    }
                }
            }
        }
    }
    for key in pg_rows.keys() {
        if !duck_rows.contains_key(key) {
            report.only_in_postgres.push((spec.table, key.to_string()));
        }
    }
}

/// Extract the join key from an Arrow column.
fn arrow_key(kind: KeyKind, array: &dyn Array, row: usize) -> Result<RowKey> {
    if array.is_null(row) {
        bail!("null join key");
    }
    match kind {
        KeyKind::MeasurementId => {
            let a = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("measurement_id is not Int64")?;
            Ok(RowKey::Id(a.value(row)))
        }
        KeyKind::CommitSha => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .context("commit_sha is not Utf8")?;
            Ok(RowKey::Sha(a.value(row).to_string()))
        }
    }
}

/// Extract one value cell from an Arrow column per its [`ReadKind`].
fn arrow_cell(array: &dyn Array, row: usize, kind: ReadKind) -> Result<CellValue> {
    if array.is_null(row) {
        return Ok(CellValue::Null);
    }
    let cell = match kind {
        ReadKind::Int32 => {
            let a = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("expected Int32")?;
            CellValue::Int(i64::from(a.value(row)))
        }
        ReadKind::Int64 => {
            let a = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("expected Int64")?;
            CellValue::Int(a.value(row))
        }
        ReadKind::Text => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .context("expected Utf8")?;
            CellValue::Text(a.value(row).to_string())
        }
        ReadKind::IntArray => {
            let a = array
                .as_any()
                .downcast_ref::<ListArray>()
                .context("expected List")?;
            let items = a.value(row);
            let ints = items
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("expected Int64 array items")?;
            let mut out = Vec::with_capacity(ints.len());
            for i in 0..ints.len() {
                if ints.is_null(i) {
                    // `all_runtimes_ns` items are declared NOT NULL; a null element
                    // means corrupt source data.
                    bail!("null element in a BIGINT[] value");
                }
                out.push(ints.value(i));
            }
            CellValue::Array(out)
        }
    };
    Ok(cell)
}

/// Extract the join key from a Postgres row (column 0).
fn pg_key(kind: KeyKind, row: &postgres::Row) -> Result<RowKey> {
    match kind {
        KeyKind::MeasurementId => Ok(RowKey::Id(
            row.try_get::<_, i64>(0).context("measurement_id")?,
        )),
        KeyKind::CommitSha => Ok(RowKey::Sha(
            row.try_get::<_, String>(0).context("commit_sha")?,
        )),
    }
}

/// Extract one value cell from a Postgres row per its [`ReadKind`].
fn pg_cell(row: &postgres::Row, idx: usize, kind: ReadKind) -> Result<CellValue> {
    let cell = match kind {
        ReadKind::Int32 => row
            .try_get::<_, Option<i32>>(idx)?
            .map_or(CellValue::Null, |v| CellValue::Int(i64::from(v))),
        ReadKind::Int64 => row
            .try_get::<_, Option<i64>>(idx)?
            .map_or(CellValue::Null, CellValue::Int),
        ReadKind::Text => row
            .try_get::<_, Option<String>>(idx)?
            .map_or(CellValue::Null, CellValue::Text),
        ReadKind::IntArray => row
            .try_get::<_, Option<Vec<i64>>>(idx)?
            .map_or(CellValue::Null, CellValue::Array),
    };
    Ok(cell)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_chart_canonicalizes() {
        assert_eq!(normalize_chart("taxi/take"), "TAXI/TAKE");
        assert_eq!(normalize_chart("TAXI/TAKE"), "TAXI/TAKE");
        assert_eq!(normalize_chart("tpc-h q1"), "TPC H Q1");
        assert_eq!(normalize_chart("tpc h q1"), "TPC H Q1");
    }

    #[test]
    fn display_query_group_handles_fan_out() {
        assert_eq!(
            display_query_group("tpch", Some("10"), "s3"),
            "TPC-H (S3) (SF=10)"
        );
        assert_eq!(
            display_query_group("tpch", Some("100"), "nvme"),
            "TPC-H (NVMe) (SF=100)"
        );
        assert_eq!(
            display_query_group("clickbench", None, "nvme"),
            "Clickbench"
        );
    }
}

#[cfg(test)]
mod value_verify_tests {
    use std::collections::BTreeMap;

    use vortex_bench_server::family;
    use vortex_bench_server::schema::COMMITS_DDL;

    use super::*;

    /// A fresh in-memory DuckDB carrying the v3 schema (the `commits` dim plus the
    /// five fact tables), mirroring the loader's test harness.
    fn in_memory_v3() -> duckdb::Connection {
        let conn = duckdb::Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch("SET TimeZone='UTC';").expect("utc");
        conn.execute_batch(COMMITS_DDL).expect("commits ddl");
        for fam in family::FAMILIES {
            conn.execute_batch(fam.schema_ddl).expect("fact ddl");
        }
        conn
    }

    fn spec(table: &str) -> &'static ValueSpec {
        VALUE_SPECS
            .iter()
            .find(|s| s.table == table)
            .expect("known table")
    }

    /// Read one table's keyed value cells from DuckDB.
    fn read(conn: &duckdb::Connection, table: &str) -> BTreeMap<RowKey, Vec<CellValue>> {
        let s = spec(table);
        let kinds = read_kinds(s).expect("read kinds");
        read_duck_table(conn, s, &kinds).expect("read duck table")
    }

    /// Re-read the (mutated) connection as the "target" side and compare against a
    /// previously-captured `source`. The comparison core is engine-agnostic, so a
    /// second DuckDB read faithfully stands in for the Postgres target here; the
    /// live PG16 end-to-end is PR-3.3's rehearsal-harness job.
    fn verify_one(
        conn: &duckdb::Connection,
        table: &str,
        source: &BTreeMap<RowKey, Vec<CellValue>>,
    ) -> PgVerifyReport {
        let target = read(conn, table);
        let mut report = PgVerifyReport::default();
        compare_table(spec(table), source, &target, &mut report);
        report
    }

    fn seed_one_of_each(duck: &duckdb::Connection) {
        duck.execute_batch(
            r#"
            INSERT INTO commits VALUES
              ('sha1', TIMESTAMPTZ '2024-01-15 12:34:56+00', 'msg', 'an', 'ae', 'cn', 'ce', 'tree1', 'http://x');
            INSERT INTO query_measurements VALUES
              (1, 'sha1', 'tpch', NULL, '1', 3, 'nvme', 'vortex', 'vortex', 1000, [1000,1100], 5, 6, 7, 8, 'x86_64-linux');
            INSERT INTO compression_times VALUES
              (2, 'sha1', 'ds', NULL, 'fmt', 'encode', 100, [1,2,3], 'x86_64-linux');
            INSERT INTO compression_sizes VALUES
              (3, 'sha1', 'ds', NULL, 'fmt', 4096);
            INSERT INTO random_access_times VALUES
              (4, 'sha1', 'ds', 'fmt', 200, [200,210], NULL);
            INSERT INTO vector_search_runs VALUES
              (5, 'sha1', 'sift', 'flat', 'f32', 0.95, 500, [500,510], 42, 1000, 64000, 3, 'x86_64-linux');
            "#,
        )
        .expect("seed one of each table");
    }

    #[test]
    fn timestamp_compares_as_epoch_microseconds() {
        // `epoch_us` must yield microseconds since the Unix epoch so the DuckDB and
        // Postgres timestamp reads are byte-for-byte comparable. 1s after the epoch
        // is exactly 1_000_000 us; 0.5s is 500_000 us.
        let duck = in_memory_v3();
        duck.execute_batch(
            r#"
            INSERT INTO commits VALUES
              ('e1', TIMESTAMPTZ '1970-01-01 00:00:01+00', 'm', NULL, NULL, NULL, NULL, 't', 'u'),
              ('e2', TIMESTAMPTZ '1970-01-01 00:00:00.5+00', 'm', NULL, NULL, NULL, NULL, 't', 'u');
            "#,
        )
        .expect("insert commits");
        let rows = read(&duck, "commits");
        // `timestamp` is the first value column (`ts("timestamp")`).
        assert_eq!(
            rows[&RowKey::Sha("e1".into())][0],
            CellValue::Int(1_000_000)
        );
        assert_eq!(rows[&RowKey::Sha("e2".into())][0], CellValue::Int(500_000));
    }

    #[test]
    fn clean_when_source_equals_target() {
        let duck = in_memory_v3();
        seed_one_of_each(&duck);
        let mut report = PgVerifyReport::default();
        for s in VALUE_SPECS {
            let source = read(&duck, s.table);
            let target = read(&duck, s.table);
            compare_table(s, &source, &target, &mut report);
        }
        assert!(report.is_clean(), "unexpected diffs:\n{report}");
    }

    #[test]
    fn detects_mutated_value_column() {
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO query_measurements VALUES \
             (1, 'sha', 'tpch', NULL, '1', 3, 'nvme', 'v', 'v', 1000, [1000], NULL, NULL, NULL, NULL, 'env');",
        )
        .expect("seed");
        let source = read(&duck, "query_measurements");
        duck.execute_batch(
            "UPDATE query_measurements SET value_ns = 999999 WHERE measurement_id = 1;",
        )
        .expect("mutate");

        let report = verify_one(&duck, "query_measurements", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        let m = &report.value_mismatches[0];
        assert_eq!(
            (m.table, m.key.as_str(), m.column),
            ("query_measurements", "1", "value_ns")
        );
        assert_eq!(m.duckdb_value, "1000");
        assert_eq!(m.pg_value, "999999");
        assert!(!report.is_clean());
    }

    #[test]
    fn detects_mutated_env_triple() {
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO compression_times VALUES (1, 'sha', 'ds', NULL, 'fmt', 'encode', 100, [1,2], 'x86_64-linux');",
        )
        .expect("seed");
        let source = read(&duck, "compression_times");
        duck.execute_batch(
            "UPDATE compression_times SET env_triple = 'aarch64-darwin' WHERE measurement_id = 1;",
        )
        .expect("mutate");

        let report = verify_one(&duck, "compression_times", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        let m = &report.value_mismatches[0];
        assert_eq!(m.column, "env_triple");
        // Text cells render quoted so an empty/whitespace value stays visible.
        assert_eq!(m.duckdb_value, "\"x86_64-linux\"");
        assert_eq!(m.pg_value, "\"aarch64-darwin\"");
    }

    #[test]
    fn detects_mutated_array_element() {
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO compression_times VALUES (1, 'sha', 'ds', NULL, 'fmt', 'encode', 100, [1,2,3], 'env');",
        )
        .expect("seed");
        let source = read(&duck, "compression_times");
        duck.execute_batch(
            "UPDATE compression_times SET all_runtimes_ns = [1,999,3] WHERE measurement_id = 1;",
        )
        .expect("mutate");

        let report = verify_one(&duck, "compression_times", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        let m = &report.value_mismatches[0];
        assert_eq!(m.column, "all_runtimes_ns");
        assert_eq!(m.duckdb_value, "[1, 2, 3]");
        assert_eq!(m.pg_value, "[1, 999, 3]");
    }

    #[test]
    fn detects_reordered_array_element_wise() {
        // Element-wise comparison is order-sensitive: a permutation is a mismatch.
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO compression_times VALUES (1, 'sha', 'ds', NULL, 'fmt', 'encode', 100, [1,2,3], 'env');",
        )
        .expect("seed");
        let source = read(&duck, "compression_times");
        duck.execute_batch(
            "UPDATE compression_times SET all_runtimes_ns = [3,2,1] WHERE measurement_id = 1;",
        )
        .expect("mutate");

        let report = verify_one(&duck, "compression_times", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        assert_eq!(report.value_mismatches[0].column, "all_runtimes_ns");
    }

    #[test]
    fn detects_mutated_vector_search_int32_and_side_counters() {
        // `vector_search_runs` is the only table with a `ReadKind::Int32` value
        // column (`iterations`) alongside the BIGINT side counters; pin the
        // mismatch-detection path for both an Int32 column and a side counter.
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO vector_search_runs VALUES \
             (7, 'sha', 'sift', 'flat', 'f32', 0.95, 500, [500,510], 42, 1000, 64000, 3, 'x86_64-linux');",
        )
        .expect("seed");
        let source = read(&duck, "vector_search_runs");
        duck.execute_batch(
            "UPDATE vector_search_runs SET iterations = 9, matches = 99 WHERE measurement_id = 7;",
        )
        .expect("mutate");

        let report = verify_one(&duck, "vector_search_runs", &source);
        assert_eq!(report.value_mismatches.len(), 2, "{report}");
        let by_col = |c: &str| {
            report
                .value_mismatches
                .iter()
                .find(|m| m.column == c)
                .unwrap_or_else(|| panic!("no mismatch for {c}"))
        };
        assert_eq!(by_col("iterations").duckdb_value, "3");
        assert_eq!(by_col("iterations").pg_value, "9");
        assert_eq!(by_col("matches").duckdb_value, "42");
        assert_eq!(by_col("matches").pg_value, "99");
    }

    #[test]
    fn value_verify_handles_empty_arrays() {
        // An empty `BIGINT[]` reads as `Array(vec![])`: it compares clean against an
        // identical empty array and mismatches against a non-empty one.
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO compression_times VALUES (1, 'sha', 'ds', NULL, 'fmt', 'encode', 100, []::BIGINT[], 'env');",
        )
        .expect("seed");
        let source = read(&duck, "compression_times");

        let mut clean = PgVerifyReport::default();
        compare_table(
            spec("compression_times"),
            &source,
            &read(&duck, "compression_times"),
            &mut clean,
        );
        assert!(clean.is_clean(), "empty-array clean case diffed:\n{clean}");

        duck.execute_batch(
            "UPDATE compression_times SET all_runtimes_ns = [1] WHERE measurement_id = 1;",
        )
        .expect("mutate");
        let report = verify_one(&duck, "compression_times", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        let m = &report.value_mismatches[0];
        assert_eq!(m.column, "all_runtimes_ns");
        assert_eq!(m.duckdb_value, "[]");
        assert_eq!(m.pg_value, "[1]");
    }

    #[test]
    fn detects_mutated_commit_metadata() {
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO commits VALUES ('s1', TIMESTAMPTZ '2024-01-01 00:00:00+00', 'msg', 'an', 'ae', 'cn', 'ce', 't1', 'u1');",
        )
        .expect("seed");
        let source = read(&duck, "commits");
        duck.execute_batch("UPDATE commits SET message = 'changed' WHERE commit_sha = 's1';")
            .expect("mutate");

        let report = verify_one(&duck, "commits", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        let m = &report.value_mismatches[0];
        assert_eq!(
            (m.table, m.key.as_str(), m.column),
            ("commits", "s1", "message")
        );
        assert_eq!(m.duckdb_value, "\"msg\"");
        assert_eq!(m.pg_value, "\"changed\"");
    }

    #[test]
    fn detects_null_versus_value() {
        // A nullable value column that is NULL on one side and set on the other is
        // a mismatch (NULL is distinct from any concrete value).
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO query_measurements VALUES \
             (1, 'sha', 'tpch', NULL, '1', 3, 'nvme', 'v', 'v', 1000, [1000], NULL, NULL, NULL, NULL, 'env');",
        )
        .expect("seed");
        let source = read(&duck, "query_measurements");
        duck.execute_batch(
            "UPDATE query_measurements SET peak_physical = 10 WHERE measurement_id = 1;",
        )
        .expect("mutate");

        let report = verify_one(&duck, "query_measurements", &source);
        assert_eq!(report.value_mismatches.len(), 1, "{report}");
        let m = &report.value_mismatches[0];
        assert_eq!(m.column, "peak_physical");
        assert_eq!(m.duckdb_value, "NULL");
        assert_eq!(m.pg_value, "10");
    }

    #[test]
    fn detects_presence_diffs_both_directions() {
        let duck = in_memory_v3();
        duck.execute_batch(
            "INSERT INTO compression_sizes VALUES (1, 'sha', 'ds', NULL, 'fmt', 4096); \
             INSERT INTO compression_sizes VALUES (2, 'sha', 'ds', NULL, 'fmt', 8192);",
        )
        .expect("seed two rows");
        let both = read(&duck, "compression_sizes");
        duck.execute_batch("DELETE FROM compression_sizes WHERE measurement_id = 2;")
            .expect("delete one");
        let one = read(&duck, "compression_sizes");

        // Source has 2 keys, target has 1: key 2 is only in DuckDB.
        let mut report = PgVerifyReport::default();
        compare_table(spec("compression_sizes"), &both, &one, &mut report);
        assert_eq!(
            report.only_in_duckdb,
            vec![("compression_sizes", "2".to_string())]
        );
        assert!(report.only_in_postgres.is_empty());
        assert!(report.value_mismatches.is_empty());

        // Reverse the roles: now key 2 is only in the Postgres target.
        let mut reverse = PgVerifyReport::default();
        compare_table(spec("compression_sizes"), &one, &both, &mut reverse);
        assert_eq!(
            reverse.only_in_postgres,
            vec![("compression_sizes", "2".to_string())]
        );
        assert!(reverse.only_in_duckdb.is_empty());
    }

    #[test]
    fn select_sql_wraps_timestamp_in_epoch_and_leaves_other_columns_bare() {
        assert_eq!(
            duck_select_sql(spec("commits")),
            "SELECT commit_sha, epoch_us(timestamp) AS timestamp, message, author_name, \
             author_email, committer_name, committer_email, tree_sha, url FROM commits"
        );
        assert_eq!(
            pg_select_sql(spec("commits")),
            "SELECT commit_sha, (extract(epoch from timestamp) * 1000000)::bigint AS timestamp, \
             message, author_name, author_email, committer_name, committer_email, tree_sha, url \
             FROM commits"
        );
        // A fact table has no timestamp column, so both reads are the bare columns.
        assert_eq!(
            duck_select_sql(spec("compression_sizes")),
            "SELECT measurement_id, value_bytes FROM compression_sizes"
        );
    }

    #[test]
    fn every_value_column_resolves_to_a_read_kind() {
        // Guards the value-column classification against the loader's column table:
        // a typo'd name or an f64 value column would fail here.
        for s in VALUE_SPECS {
            read_kinds(s).unwrap_or_else(|e| panic!("{}: {e}", s.table));
        }
    }

    #[test]
    fn report_display_lists_diffs_and_clean_summary() {
        let clean = PgVerifyReport::default();
        assert!(clean.is_clean());
        assert!(clean.to_string().contains("0 value mismatches"));

        let mut dirty = PgVerifyReport::default();
        dirty
            .only_in_duckdb
            .push(("query_measurements", "7".to_string()));
        dirty.value_mismatches.push(ValueMismatch {
            table: "commits",
            key: "s1".to_string(),
            column: "message",
            duckdb_value: "\"a\"".to_string(),
            pg_value: "\"b\"".to_string(),
        });
        let rendered = dirty.to_string();
        assert!(!dirty.is_clean());
        assert!(rendered.contains("query_measurements: 7"), "{rendered}");
        assert!(
            rendered.contains("commits [s1] message: duckdb=\"a\" pg=\"b\""),
            "{rendered}"
        );
    }
}
