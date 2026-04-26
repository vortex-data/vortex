// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end migration of one v2 dataset into a v3 DuckDB file.
//!
//! Streams `data.json.gz` line-by-line, runs each record through the
//! [classifier][crate::classifier], and writes one row per record into
//! the appropriate v3 fact table. Every row's `measurement_id` is
//! computed via the server's `measurement_id_*` functions so the result
//! is byte-compatible with what fresh `/api/ingest` would have produced.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use duckdb::Transaction;
use duckdb::params;
use tracing::warn;
use vortex_bench_server::db::measurement_id_compression_size;
use vortex_bench_server::db::measurement_id_compression_time;
use vortex_bench_server::db::measurement_id_query;
use vortex_bench_server::db::measurement_id_random_access;
use vortex_bench_server::records::CompressionSize;
use vortex_bench_server::records::CompressionTime;
use vortex_bench_server::records::QueryMeasurement;
use vortex_bench_server::records::RandomAccessTime;
use vortex_bench_server::schema::SCHEMA_DDL;

use crate::classifier::V3Bin;
use crate::classifier::classify;
use crate::commits::upsert_commit;
use crate::source::Source;
use crate::v2::V2Commit;
use crate::v2::V2FileSize;
use crate::v2::V2Record;
use crate::v2::index_commits;
use crate::v2::runtime_as_i64;
use crate::v2::value_as_f64;

/// Per-table insert counts, plus skip / missing counts.
#[derive(Debug, Default, Clone)]
pub struct MigrationSummary {
    pub records_read: u64,
    pub query_inserted: u64,
    pub compression_time_inserted: u64,
    pub compression_size_inserted: u64,
    pub random_access_inserted: u64,
    pub file_size_inserted: u64,
    pub uncategorized: u64,
    pub uncategorized_prefixes: BTreeMap<String, u64>,
    pub missing_commit: u64,
    pub commit_warnings: u64,
    pub skipped_no_value: u64,
    pub commits_inserted: u64,
}

impl MigrationSummary {
    /// Total `data.json.gz` records that landed in some v3 fact table.
    pub fn total_inserted(&self) -> u64 {
        self.query_inserted
            + self.compression_time_inserted
            + self.compression_size_inserted
            + self.random_access_inserted
    }

    /// Fraction of records that were uncategorized. The orchestrator
    /// stops if this exceeds the documented 5% threshold.
    pub fn uncategorized_fraction(&self) -> f64 {
        if self.records_read == 0 {
            return 0.0;
        }
        self.uncategorized as f64 / self.records_read as f64
    }
}

/// Open or create a DuckDB at `path` and apply the v3 schema.
pub fn open_target_db(path: &Path) -> Result<Connection> {
    let conn =
        Connection::open(path).with_context(|| format!("opening DuckDB at {}", path.display()))?;
    conn.execute_batch(SCHEMA_DDL)
        .context("applying v3 schema DDL")?;
    Ok(conn)
}

/// Run the whole migration: commits, data.json.gz, and every
/// file-sizes-*.json.gz under the source.
pub fn run(source: &Source, target: &Path) -> Result<MigrationSummary> {
    let mut conn = open_target_db(target)?;
    let mut summary = MigrationSummary::default();

    let commits = read_commits(source)?;
    summary.commits_inserted = upsert_all_commits(&mut conn, &commits, &mut summary)?;

    migrate_data_jsonl(&mut conn, source, &commits, &mut summary)?;

    for name in source.list_file_sizes()? {
        if let Err(e) = migrate_file_sizes(&mut conn, source, &name, &commits, &mut summary) {
            warn!("file-sizes file {name} failed: {e:#}");
        }
    }

    Ok(summary)
}

fn read_commits(source: &Source) -> Result<BTreeMap<String, V2Commit>> {
    let reader = source.open_commits_jsonl()?;
    let mut commits: Vec<V2Commit> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<V2Commit>(trimmed) {
            Ok(c) => commits.push(c),
            Err(e) => warn!("skipping malformed commits.json line: {e}"),
        }
    }
    Ok(index_commits(commits))
}

fn upsert_all_commits(
    conn: &mut Connection,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
) -> Result<u64> {
    let tx = conn.transaction().context("begin commits transaction")?;
    let mut count = 0u64;
    for commit in commits.values() {
        let outcome = upsert_commit(&tx, commit)?;
        for w in outcome.warnings {
            warn!("{w}");
            summary.commit_warnings += 1;
        }
        count += 1;
    }
    tx.commit().context("commit commits transaction")?;
    Ok(count)
}

fn migrate_data_jsonl(
    conn: &mut Connection,
    source: &Source,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
) -> Result<()> {
    let reader = source.open_data_jsonl()?;
    let mut tx = conn.transaction().context("begin data tx")?;
    const BATCH: u64 = 10_000;
    let mut in_batch = 0u64;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        summary.records_read += 1;
        let record: V2Record = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                warn!("skipping malformed data.json line: {e}");
                continue;
            }
        };
        apply_v2_record(&tx, &record, commits, summary)?;
        in_batch += 1;
        if in_batch >= BATCH {
            tx.commit().context("commit data batch")?;
            tx = conn.transaction().context("begin data tx")?;
            in_batch = 0;
        }
    }
    tx.commit().context("commit final data batch")?;
    Ok(())
}

fn apply_v2_record(
    tx: &Transaction<'_>,
    record: &V2Record,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
) -> Result<()> {
    let Some(sha) = record.commit_id.clone() else {
        summary.missing_commit += 1;
        return Ok(());
    };
    if !commits.contains_key(&sha) {
        summary.missing_commit += 1;
        return Ok(());
    }

    let Some(bin) = classify(record) else {
        summary.uncategorized += 1;
        let prefix = record.name.split('/').next().unwrap_or("").to_string();
        *summary.uncategorized_prefixes.entry(prefix).or_insert(0) += 1;
        return Ok(());
    };

    let env_triple = record.env_triple.as_ref().and_then(|t| t.to_triple());
    let runtimes = record
        .all_runtimes
        .as_ref()
        .map(|v| v.iter().filter_map(runtime_as_i64).collect::<Vec<i64>>())
        .unwrap_or_default();
    let value_f64 = match record.value.as_ref().and_then(value_as_f64) {
        Some(v) => v,
        None => {
            summary.skipped_no_value += 1;
            return Ok(());
        }
    };

    match bin {
        V3Bin::Query {
            dataset,
            dataset_variant,
            scale_factor,
            query_idx,
            storage,
            engine,
            format,
        } => {
            let qm = QueryMeasurement {
                commit_sha: sha,
                dataset,
                dataset_variant,
                scale_factor,
                query_idx,
                storage,
                engine,
                format,
                value_ns: value_f64 as i64,
                all_runtimes_ns: runtimes,
                peak_physical: None,
                peak_virtual: None,
                physical_delta: None,
                virtual_delta: None,
                env_triple,
            };
            insert_query(tx, &qm)?;
            summary.query_inserted += 1;
        }
        V3Bin::CompressionTime {
            dataset,
            dataset_variant,
            format,
            op,
        } => {
            let ct = CompressionTime {
                commit_sha: sha,
                dataset,
                dataset_variant,
                format,
                op,
                value_ns: value_f64 as i64,
                all_runtimes_ns: runtimes,
                env_triple,
            };
            insert_compression_time(tx, &ct)?;
            summary.compression_time_inserted += 1;
        }
        V3Bin::CompressionSize {
            dataset,
            dataset_variant,
            format,
        } => {
            let cs = CompressionSize {
                commit_sha: sha,
                dataset,
                dataset_variant,
                format,
                value_bytes: value_f64 as i64,
            };
            insert_compression_size(tx, &cs)?;
            summary.compression_size_inserted += 1;
        }
        V3Bin::RandomAccess { dataset, format } => {
            let ra = RandomAccessTime {
                commit_sha: sha,
                dataset,
                format,
                value_ns: value_f64 as i64,
                all_runtimes_ns: runtimes,
                env_triple,
            };
            insert_random_access(tx, &ra)?;
            summary.random_access_inserted += 1;
        }
    }
    Ok(())
}

fn insert_query(tx: &Transaction<'_>, r: &QueryMeasurement) -> Result<()> {
    let mid = measurement_id_query(r);
    tx.execute(
        r#"
        INSERT INTO query_measurements (
            measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
            query_idx, storage, engine, format,
            value_ns, all_runtimes_ns,
            peak_physical, peak_virtual, physical_delta, virtual_delta,
            env_triple
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?, ?, ?, ?, ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            env_triple      = excluded.env_triple
        "#,
        params![
            mid,
            r.commit_sha,
            r.dataset,
            r.dataset_variant,
            r.scale_factor,
            r.query_idx,
            r.storage,
            r.engine,
            r.format,
            r.value_ns,
            runtimes_literal(&r.all_runtimes_ns),
            r.peak_physical,
            r.peak_virtual,
            r.physical_delta,
            r.virtual_delta,
            r.env_triple,
        ],
    )?;
    Ok(())
}

fn insert_compression_time(tx: &Transaction<'_>, r: &CompressionTime) -> Result<()> {
    let mid = measurement_id_compression_time(r);
    tx.execute(
        r#"
        INSERT INTO compression_times (
            measurement_id, commit_sha, dataset, dataset_variant,
            format, op, value_ns, all_runtimes_ns, env_triple
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            env_triple      = excluded.env_triple
        "#,
        params![
            mid,
            r.commit_sha,
            r.dataset,
            r.dataset_variant,
            r.format,
            r.op,
            r.value_ns,
            runtimes_literal(&r.all_runtimes_ns),
            r.env_triple,
        ],
    )?;
    Ok(())
}

fn insert_compression_size(tx: &Transaction<'_>, r: &CompressionSize) -> Result<()> {
    let mid = measurement_id_compression_size(r);
    tx.execute(
        r#"
        INSERT INTO compression_sizes (
            measurement_id, commit_sha, dataset, dataset_variant,
            format, value_bytes
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha   = excluded.commit_sha,
            value_bytes  = excluded.value_bytes
        "#,
        params![
            mid,
            r.commit_sha,
            r.dataset,
            r.dataset_variant,
            r.format,
            r.value_bytes,
        ],
    )?;
    Ok(())
}

fn insert_random_access(tx: &Transaction<'_>, r: &RandomAccessTime) -> Result<()> {
    let mid = measurement_id_random_access(r);
    tx.execute(
        r#"
        INSERT INTO random_access_times (
            measurement_id, commit_sha, dataset, format,
            value_ns, all_runtimes_ns, env_triple
        ) VALUES (?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            env_triple      = excluded.env_triple
        "#,
        params![
            mid,
            r.commit_sha,
            r.dataset,
            r.format,
            r.value_ns,
            runtimes_literal(&r.all_runtimes_ns),
            r.env_triple,
        ],
    )?;
    Ok(())
}

fn runtimes_literal(values: &[i64]) -> String {
    let mut s = String::with_capacity(values.len() * 8 + 2);
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
    s
}

fn migrate_file_sizes(
    conn: &mut Connection,
    source: &Source,
    name: &str,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
) -> Result<()> {
    let reader = source.open_file_sizes(name)?;
    let dataset = name
        .strip_prefix("file-sizes-")
        .and_then(|s| s.strip_suffix(".json.gz"))
        .unwrap_or(name)
        .to_string();
    let mut tx = conn.transaction().context("begin file-sizes tx")?;
    const BATCH: u64 = 10_000;
    let mut in_batch = 0u64;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let sz: V2FileSize = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                warn!("skipping malformed {name} line: {e}");
                continue;
            }
        };
        if !commits.contains_key(&sz.commit_id) {
            summary.missing_commit += 1;
            continue;
        }
        // file-sizes-*.json.gz captures per-file sizes inside one
        // benchmark/format/scale_factor combo. We aggregate to one
        // (commit, dataset, dataset_variant, format) row by summing,
        // since v3's compression_sizes is a single bytes value per
        // (dim) tuple. Use ON CONFLICT to accumulate.
        upsert_file_size_row(&tx, &sz, &dataset)?;
        summary.file_size_inserted += 1;
        in_batch += 1;
        if in_batch >= BATCH {
            tx.commit().context("commit file-sizes batch")?;
            tx = conn.transaction().context("begin file-sizes tx")?;
            in_batch = 0;
        }
    }
    tx.commit().context("commit final file-sizes batch")?;
    Ok(())
}

fn upsert_file_size_row(
    tx: &Transaction<'_>,
    sz: &V2FileSize,
    dataset_fallback: &str,
) -> Result<()> {
    let dataset = if sz.benchmark.is_empty() {
        dataset_fallback.to_string()
    } else {
        sz.benchmark.clone()
    };
    let dataset_variant = sz
        .scale_factor
        .as_ref()
        .filter(|s| !s.is_empty() && s.as_str() != "1.0")
        .cloned();
    let cs = CompressionSize {
        commit_sha: sz.commit_id.clone(),
        dataset,
        dataset_variant,
        format: sz.format.clone(),
        value_bytes: sz.size_bytes,
    };
    let mid = measurement_id_compression_size(&cs);
    // Multiple files within the same dataset/format/scale_factor sum
    // into one row by adding to whatever is already there.
    tx.execute(
        r#"
        INSERT INTO compression_sizes (
            measurement_id, commit_sha, dataset, dataset_variant,
            format, value_bytes
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            value_bytes = compression_sizes.value_bytes + excluded.value_bytes
        "#,
        params![
            mid,
            cs.commit_sha,
            cs.dataset,
            cs.dataset_variant,
            cs.format,
            cs.value_bytes,
        ],
    )?;
    Ok(())
}

/// Print the summary in a human-readable form. Returned by the CLI.
impl std::fmt::Display for MigrationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Records read:           {}", self.records_read)?;
        writeln!(f, "Commits upserted:       {}", self.commits_inserted)?;
        writeln!(f, "Commit warnings:        {}", self.commit_warnings)?;
        writeln!(f, "Inserted (query):       {}", self.query_inserted)?;
        writeln!(
            f,
            "Inserted (compress t):  {}",
            self.compression_time_inserted
        )?;
        writeln!(
            f,
            "Inserted (compress s):  {}",
            self.compression_size_inserted
        )?;
        writeln!(f, "Inserted (random acc):  {}", self.random_access_inserted)?;
        writeln!(f, "Inserted (file sizes):  {}", self.file_size_inserted)?;
        writeln!(f, "Missing commit:         {}", self.missing_commit)?;
        writeln!(f, "Skipped (no value):     {}", self.skipped_no_value)?;
        writeln!(
            f,
            "Uncategorized:          {} ({:.2}%)",
            self.uncategorized,
            100.0 * self.uncategorized_fraction()
        )?;
        if !self.uncategorized_prefixes.is_empty() {
            let mut top: Vec<_> = self.uncategorized_prefixes.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            writeln!(f, "Top uncategorized prefixes:")?;
            for (prefix, n) in top.iter().take(20) {
                writeln!(f, "  {prefix:>32} : {n}")?;
            }
        }
        Ok(())
    }
}
