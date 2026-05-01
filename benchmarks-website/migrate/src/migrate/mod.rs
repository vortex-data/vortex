// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end migration of one v2 dataset into a v3 DuckDB file.
//!
//! Streams `data.json.gz` line-by-line, runs each record through the
//! [`crate::classifier`], and writes one row per record into the appropriate
//! v3 fact table. Every row's `measurement_id` is computed via the server's
//! `measurement_id_*` functions so the result is byte-compatible with what
//! fresh `/api/ingest` would have produced.
//!
//! Bulk-load shape: rows are accumulated in memory as parallel column
//! vectors, deduplicated by `measurement_id`, then flushed to DuckDB
//! via `Appender::append_record_batch` as one Arrow `RecordBatch` per
//! fact table.

mod accum;

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use arrow_array::RecordBatch;
use duckdb::Connection;
use tracing::info;
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

use self::accum::CompressionSizeAccum;
use self::accum::CompressionTimeAccum;
use self::accum::QueryAccum;
use self::accum::RandomAccessAccum;
use self::accum::build_compression_size_batch;
use self::accum::build_compression_time_batch;
use self::accum::build_query_batch;
use self::accum::build_random_access_batch;
use crate::classifier;
use crate::classifier::V3Bin;
use crate::commits::upsert_commit;
use crate::source::KNOWN_FILE_SIZES_SUITES;
use crate::source::Source;
use crate::v2::V2Commit;
use crate::v2::V2FileSize;
use crate::v2::V2Record;
use crate::v2::canonical_scale_factor;
use crate::v2::index_commits;
use crate::v2::runtime_as_i64;
use crate::v2::value_as_f64;

/// Per-table insert counts, plus skip / missing counts.
#[derive(Debug, Default, Clone)]
pub struct MigrationSummary {
    /// Lines read from `data.json.gz`.
    pub records_read: u64,
    /// Rows successfully inserted into `query_measurements`.
    pub query_inserted: u64,
    /// Rows successfully inserted into `compression_times`.
    pub compression_time_inserted: u64,
    /// Rows successfully inserted into `compression_sizes`.
    pub compression_size_inserted: u64,
    /// Rows successfully inserted into `random_access_times`.
    pub random_access_inserted: u64,
    /// `file-sizes-*.json.gz` lines folded into `compression_sizes`.
    pub file_size_inserted: u64,
    /// Records the classifier returned `Unknown` for.
    pub uncategorized: u64,
    /// Top-level prefix histogram of uncategorised records, for triage.
    pub uncategorized_prefixes: BTreeMap<String, u64>,
    /// Records whose `commit_id` doesn't match any commit in `commits.jsonl`.
    pub missing_commit: u64,
    /// Warnings emitted while upserting commits (e.g. missing tree SHA).
    pub commit_warnings: u64,
    /// Records dropped because their `value` was missing or non-numeric.
    pub skipped_no_value: u64,
    /// Records the classifier returned `Skip(reason)` for.
    pub skipped_intentional: u64,
    /// Commits upserted into the `commits` dim table.
    pub commits_inserted: u64,
    /// Records dropped by dedup because their `measurement_id` collided
    /// with a previously kept row.
    pub deduped: u64,
    /// Number of records dropped by dedup whose `value_ns` (or
    /// `value_bytes` for compression_sizes' replace path) differed
    /// from the kept row's. Non-zero is a smell worth investigating.
    pub deduped_with_conflict: u64,
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

/// Open or create a DuckDB at `path` and apply the v3 schema. The
/// migrator is a one-shot fresh load; the bulk-append flush is pure
/// insert (no `ON CONFLICT`), so any stale rows in `path` would clash
/// with the next run on the same primary keys. Delete both the
/// database file and its WAL companion up front so every run starts
/// from a known-empty state.
pub fn open_target_db(path: &Path) -> Result<Connection> {
    remove_if_exists(path)?;
    let wal = wal_path(path);
    remove_if_exists(&wal)?;
    let conn =
        Connection::open(path).with_context(|| format!("opening DuckDB at {}", path.display()))?;
    conn.execute_batch(SCHEMA_DDL)
        .context("applying v3 schema DDL")?;
    Ok(conn)
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => {
            info!(path = %path.display(), "removed pre-existing target file");
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("removing {}", path.display())),
    }
}

/// DuckDB writes its write-ahead log next to the database file with a
/// `.wal` suffix appended (e.g. `v3.duckdb` -> `v3.duckdb.wal`).
fn wal_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".wal");
    std::path::PathBuf::from(name)
}

/// Run the whole migration: commits, data.json.gz, and every
/// file-sizes-*.json.gz under the source.
pub fn run(source: &Source, target: &Path) -> Result<MigrationSummary> {
    let mut conn = open_target_db(target)?;
    let mut summary = MigrationSummary::default();

    info!(source = %source.describe(), "Reading commits.json");
    let commits = read_commits(source)?;
    info!(commits = commits.len(), "Loaded commits");
    summary.commits_inserted = upsert_all_commits(&mut conn, &commits, &mut summary)?;

    let mut q = QueryAccum::default();
    let mut ct = CompressionTimeAccum::default();
    let mut cs = CompressionSizeAccum::default();
    let mut ra = RandomAccessAccum::default();

    info!("Migrating data.json.gz");
    migrate_data_jsonl(
        source,
        &commits,
        &mut summary,
        &mut q,
        &mut ct,
        &mut cs,
        &mut ra,
    )?;
    info!(records = summary.records_read, "data.json.gz done");

    for name in source.list_file_sizes()? {
        info!(name = %name, "Migrating file-sizes");
        if let Err(e) = migrate_file_sizes(source, &name, &commits, &mut summary, &mut cs) {
            warn!("file-sizes file {name} failed: {e:#}");
        }
    }

    info!("Flushing accumulators to DuckDB");
    flush_all(&conn, q, ct, ra, cs, &mut summary)?;

    Ok(summary)
}

/// Flush each accumulator's batch and bump the matching per-fact
/// summary counter only AFTER the flush succeeds. This way a flush
/// failure leaves the counter at zero (or its previous value) rather
/// than reporting rows that never landed in DuckDB.
fn flush_all(
    conn: &Connection,
    q: QueryAccum,
    ct: CompressionTimeAccum,
    ra: RandomAccessAccum,
    cs: CompressionSizeAccum,
    summary: &mut MigrationSummary,
) -> Result<()> {
    let batch = build_query_batch(q)?;
    let n = batch.num_rows() as u64;
    flush(conn, "query_measurements", batch)?;
    summary.query_inserted = n;

    let batch = build_compression_time_batch(ct)?;
    let n = batch.num_rows() as u64;
    flush(conn, "compression_times", batch)?;
    summary.compression_time_inserted = n;

    let batch = build_random_access_batch(ra)?;
    let n = batch.num_rows() as u64;
    flush(conn, "random_access_times", batch)?;
    summary.random_access_inserted = n;

    let batch = build_compression_size_batch(cs)?;
    let n = batch.num_rows() as u64;
    flush(conn, "compression_sizes", batch)?;
    summary.compression_size_inserted = n;

    Ok(())
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

/// Stream `data.json.gz` and push classified records into the
/// per-table accumulators. Dedup happens inside each accumulator's
/// `push` method by `measurement_id`.
fn migrate_data_jsonl(
    source: &Source,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
    q: &mut QueryAccum,
    ct: &mut CompressionTimeAccum,
    cs: &mut CompressionSizeAccum,
    ra: &mut RandomAccessAccum,
) -> Result<()> {
    let reader = source.open_data_jsonl()?;
    let started = Instant::now();
    let mut last_log = Instant::now();
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
        apply_v2_record(&record, commits, summary, q, ct, cs, ra);
        if last_log.elapsed() >= Duration::from_secs(5) {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = summary.records_read as f64 / elapsed.max(0.001);
            info!(
                records = summary.records_read,
                rate = format!("{rate:.0}/s"),
                query = q.measurement_id.len(),
                compression_time = ct.measurement_id.len(),
                compression_size = cs.rows.len(),
                random_access = ra.measurement_id.len(),
                "migration progress",
            );
            last_log = Instant::now();
        }
    }
    Ok(())
}

fn apply_v2_record(
    record: &V2Record,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
    q: &mut QueryAccum,
    ct: &mut CompressionTimeAccum,
    cs: &mut CompressionSizeAccum,
    ra: &mut RandomAccessAccum,
) {
    let Some(sha) = record.commit_id.clone() else {
        summary.missing_commit += 1;
        return;
    };
    if !commits.contains_key(&sha) {
        summary.missing_commit += 1;
        return;
    }

    let bin = match classifier::classify_outcome(record) {
        classifier::Outcome::Bin(b) => b,
        classifier::Outcome::Skip(_) => {
            summary.skipped_intentional += 1;
            return;
        }
        classifier::Outcome::Unknown => {
            summary.uncategorized += 1;
            let prefix = record.name.split('/').next().unwrap_or("").to_string();
            *summary.uncategorized_prefixes.entry(prefix).or_insert(0) += 1;
            return;
        }
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
            return;
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
            let mid = measurement_id_query(&qm);
            q.push(mid, qm, summary);
        }
        V3Bin::CompressionTime {
            dataset,
            dataset_variant,
            format,
            op,
        } => {
            let ctr = CompressionTime {
                commit_sha: sha,
                dataset,
                dataset_variant,
                format,
                op,
                value_ns: value_f64 as i64,
                all_runtimes_ns: runtimes,
                env_triple,
            };
            let mid = measurement_id_compression_time(&ctr);
            ct.push(mid, ctr, summary);
        }
        V3Bin::CompressionSize {
            dataset,
            dataset_variant,
            format,
        } => {
            let csr = CompressionSize {
                commit_sha: sha,
                dataset,
                dataset_variant,
                format,
                value_bytes: value_f64 as i64,
            };
            let mid = measurement_id_compression_size(&csr);
            cs.push_replace(mid, csr, summary);
        }
        V3Bin::RandomAccess { dataset, format } => {
            let rar = RandomAccessTime {
                commit_sha: sha,
                dataset,
                format,
                value_ns: value_f64 as i64,
                all_runtimes_ns: runtimes,
                env_triple,
            };
            let mid = measurement_id_random_access(&rar);
            ra.push(mid, rar, summary);
        }
    }
}

fn migrate_file_sizes(
    source: &Source,
    name: &str,
    commits: &BTreeMap<String, V2Commit>,
    summary: &mut MigrationSummary,
    cs: &mut CompressionSizeAccum,
) -> Result<()> {
    let reader = source.open_file_sizes(name)?;
    // Prefix unknown-id fallbacks with `unknown:` so they're clearly
    // labeled in the UI rather than masquerading as a dataset name.
    let dataset_fallback = {
        let stripped = name
            .strip_prefix("file-sizes-")
            .and_then(|s| s.strip_suffix(".json.gz"))
            .unwrap_or(name);
        if KNOWN_FILE_SIZES_SUITES.contains(&stripped) {
            stripped.to_string()
        } else {
            format!("unknown:{stripped}")
        }
    };
    let started = Instant::now();
    let mut last_log = Instant::now();
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
        let dataset = if sz.benchmark.is_empty() {
            dataset_fallback.clone()
        } else {
            sz.benchmark.clone()
        };
        // Run SF through canonical_scale_factor so `"1"`, `"1.0"`, `"10"`
        // and `"10.0"` collapse to one form, matching what
        // `bin_compression_size` writes for the data.json.gz path.
        let dataset_variant = canonical_scale_factor(sz.scale_factor.as_deref());
        let csr = CompressionSize {
            commit_sha: sz.commit_id.clone(),
            dataset,
            dataset_variant,
            format: sz.format.clone(),
            value_bytes: sz.size_bytes,
        };
        let mid = measurement_id_compression_size(&csr);
        cs.push_sum(mid, csr);
        summary.file_size_inserted += 1;
        if last_log.elapsed() >= Duration::from_secs(5) {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = summary.file_size_inserted as f64 / elapsed.max(0.001);
            info!(
                name = %name,
                file_sizes = summary.file_size_inserted,
                rate = format!("{rate:.0}/s"),
                "file-sizes progress",
            );
            last_log = Instant::now();
        }
    }
    Ok(())
}

/// Append an Arrow `RecordBatch` to a DuckDB table via `Appender`.
fn flush(conn: &Connection, table: &str, batch: RecordBatch) -> Result<()> {
    let mut app = conn
        .appender(table)
        .with_context(|| format!("opening appender for {table}"))?;
    app.append_record_batch(batch)
        .with_context(|| format!("appending record batch to {table}"))?;
    drop(app);
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
        writeln!(f, "Skipped (intentional):  {}", self.skipped_intentional)?;
        writeln!(f, "Deduplicated:           {}", self.deduped)?;
        writeln!(f, "Dedup w/ value diff:    {}", self.deduped_with_conflict)?;
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

#[cfg(test)]
mod tests {
    use vortex_bench_server::records::QueryMeasurement;

    use super::*;

    fn open_db_without(table: &str) -> Result<(tempfile::TempDir, Connection)> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path().join("v3.duckdb");
        let conn = open_target_db(&path)?;
        conn.execute_batch(&format!("DROP TABLE {table}"))?;
        Ok((dir, conn))
    }

    fn one_query_row() -> QueryMeasurement {
        QueryMeasurement {
            commit_sha: "deadbeef".into(),
            dataset: "clickbench".into(),
            dataset_variant: None,
            scale_factor: None,
            query_idx: 7,
            storage: "nvme".into(),
            engine: "datafusion".into(),
            format: "parquet".into(),
            value_ns: 100,
            all_runtimes_ns: vec![100],
            peak_physical: None,
            peak_virtual: None,
            physical_delta: None,
            virtual_delta: None,
            env_triple: None,
        }
    }

    #[test]
    fn flush_all_does_not_overcount_on_failure() -> Result<()> {
        // Drop `compression_times` before flushing so the second
        // flush in `flush_all` fails. The first (queries) succeeded,
        // so its counter must be set; the failed table's counter and
        // every later table's counter must stay at zero.
        let (_dir, conn) = open_db_without("compression_times")?;

        let mut summary = MigrationSummary::default();
        let mut q = QueryAccum::default();
        let qm = one_query_row();
        let mid = vortex_bench_server::db::measurement_id_query(&qm);
        q.push(mid, qm, &mut summary);

        let ct = CompressionTimeAccum::default();
        let ra = RandomAccessAccum::default();
        let cs = CompressionSizeAccum::default();

        let result = flush_all(&conn, q, ct, ra, cs, &mut summary);
        assert!(result.is_err(), "expected flush to fail on missing table");

        assert_eq!(
            summary.query_inserted, 1,
            "query flushed before the failure must be counted"
        );
        assert_eq!(
            summary.compression_time_inserted, 0,
            "failed flush must not bump the counter"
        );
        assert_eq!(summary.random_access_inserted, 0, "later flushes never ran");
        assert_eq!(
            summary.compression_size_inserted, 0,
            "later flushes never ran"
        );
        Ok(())
    }
}
