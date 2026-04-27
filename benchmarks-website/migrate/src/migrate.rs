// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end migration of one v2 dataset into a v3 DuckDB file.
//!
//! Streams `data.json.gz` line-by-line, runs each record through the
//! [`classifier`], and writes one row per record into the appropriate v3 fact table.
//! Every row's `measurement_id` is computed via the server's `measurement_id_*` functions so the
//! result is byte-compatible with what fresh `/api/ingest` would have produced.
//!
//! Bulk-load shape: rows are accumulated in memory as parallel column
//! vectors, deduplicated by `measurement_id`, then flushed to DuckDB
//! via `Appender::append_record_batch` as one Arrow `RecordBatch` per
//! fact table.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
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
use vortex_utils::aliases::hash_map::HashMap;

use crate::classifier;
use crate::classifier::V3Bin;
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
    pub skipped_intentional: u64,
    pub commits_inserted: u64,
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
    summary.query_inserted = q.measurement_id.len() as u64;
    summary.compression_time_inserted = ct.measurement_id.len() as u64;
    summary.random_access_inserted = ra.measurement_id.len() as u64;
    summary.compression_size_inserted = cs.rows.len() as u64;

    flush(&conn, "query_measurements", build_query_batch(q)?)?;
    flush(
        &conn,
        "compression_times",
        build_compression_time_batch(ct)?,
    )?;
    flush(&conn, "random_access_times", build_random_access_batch(ra)?)?;
    flush(
        &conn,
        "compression_sizes",
        build_compression_size_batch(cs)?,
    )?;

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
    let dataset_fallback = name
        .strip_prefix("file-sizes-")
        .and_then(|s| s.strip_suffix(".json.gz"))
        .unwrap_or(name)
        .to_string();
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
        let dataset_variant = sz
            .scale_factor
            .as_ref()
            .filter(|s| !s.is_empty() && s.as_str() != "1.0")
            .cloned();
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

#[derive(Default)]
struct QueryAccum {
    measurement_id: Vec<i64>,
    commit_sha: Vec<String>,
    dataset: Vec<String>,
    dataset_variant: Vec<Option<String>>,
    scale_factor: Vec<Option<String>>,
    query_idx: Vec<i32>,
    storage: Vec<String>,
    engine: Vec<String>,
    format: Vec<String>,
    value_ns: Vec<i64>,
    all_runtimes_ns: Vec<Vec<i64>>,
    peak_physical: Vec<Option<i64>>,
    peak_virtual: Vec<Option<i64>>,
    physical_delta: Vec<Option<i64>>,
    virtual_delta: Vec<Option<i64>>,
    env_triple: Vec<Option<String>>,
    /// `mid` -> index in the parallel column vecs. Lets us look up the
    /// kept row's `value_ns` on collision so we can flag conflicts.
    seen: HashMap<i64, usize>,
}

impl QueryAccum {
    fn push(&mut self, mid: i64, r: QueryMeasurement, summary: &mut MigrationSummary) {
        if let Some(&idx) = self.seen.get(&mid) {
            summary.deduped += 1;
            if self.value_ns[idx] != r.value_ns {
                summary.deduped_with_conflict += 1;
            }
            return;
        }
        let idx = self.measurement_id.len();
        self.seen.insert(mid, idx);
        self.measurement_id.push(mid);
        self.commit_sha.push(r.commit_sha);
        self.dataset.push(r.dataset);
        self.dataset_variant.push(r.dataset_variant);
        self.scale_factor.push(r.scale_factor);
        self.query_idx.push(r.query_idx);
        self.storage.push(r.storage);
        self.engine.push(r.engine);
        self.format.push(r.format);
        self.value_ns.push(r.value_ns);
        self.all_runtimes_ns.push(r.all_runtimes_ns);
        self.peak_physical.push(r.peak_physical);
        self.peak_virtual.push(r.peak_virtual);
        self.physical_delta.push(r.physical_delta);
        self.virtual_delta.push(r.virtual_delta);
        self.env_triple.push(r.env_triple);
    }
}

#[derive(Default)]
struct CompressionTimeAccum {
    measurement_id: Vec<i64>,
    commit_sha: Vec<String>,
    dataset: Vec<String>,
    dataset_variant: Vec<Option<String>>,
    format: Vec<String>,
    op: Vec<String>,
    value_ns: Vec<i64>,
    all_runtimes_ns: Vec<Vec<i64>>,
    env_triple: Vec<Option<String>>,
    seen: HashMap<i64, usize>,
}

impl CompressionTimeAccum {
    fn push(&mut self, mid: i64, r: CompressionTime, summary: &mut MigrationSummary) {
        if let Some(&idx) = self.seen.get(&mid) {
            summary.deduped += 1;
            if self.value_ns[idx] != r.value_ns {
                summary.deduped_with_conflict += 1;
            }
            return;
        }
        let idx = self.measurement_id.len();
        self.seen.insert(mid, idx);
        self.measurement_id.push(mid);
        self.commit_sha.push(r.commit_sha);
        self.dataset.push(r.dataset);
        self.dataset_variant.push(r.dataset_variant);
        self.format.push(r.format);
        self.op.push(r.op);
        self.value_ns.push(r.value_ns);
        self.all_runtimes_ns.push(r.all_runtimes_ns);
        self.env_triple.push(r.env_triple);
    }
}

#[derive(Default)]
struct RandomAccessAccum {
    measurement_id: Vec<i64>,
    commit_sha: Vec<String>,
    dataset: Vec<String>,
    format: Vec<String>,
    value_ns: Vec<i64>,
    all_runtimes_ns: Vec<Vec<i64>>,
    env_triple: Vec<Option<String>>,
    seen: HashMap<i64, usize>,
}

impl RandomAccessAccum {
    fn push(&mut self, mid: i64, r: RandomAccessTime, summary: &mut MigrationSummary) {
        if let Some(&idx) = self.seen.get(&mid) {
            summary.deduped += 1;
            if self.value_ns[idx] != r.value_ns {
                summary.deduped_with_conflict += 1;
            }
            return;
        }
        let idx = self.measurement_id.len();
        self.seen.insert(mid, idx);
        self.measurement_id.push(mid);
        self.commit_sha.push(r.commit_sha);
        self.dataset.push(r.dataset);
        self.format.push(r.format);
        self.value_ns.push(r.value_ns);
        self.all_runtimes_ns.push(r.all_runtimes_ns);
        self.env_triple.push(r.env_triple);
    }
}

/// `compression_sizes` is fed by both data.json.gz (replace-on-collision)
/// and file-sizes-*.json.gz (sum-on-collision). Stored as a map; converted
/// to a `RecordBatch` at flush time.
#[derive(Default)]
struct CompressionSizeAccum {
    rows: HashMap<i64, CompressionSize>,
}

impl CompressionSizeAccum {
    /// data.json.gz path: latest write wins, mirroring the prior
    /// `ON CONFLICT DO UPDATE SET value_bytes = excluded.value_bytes`.
    /// Bumps `deduped_with_conflict` when an existing row's
    /// `value_bytes` differs from the incoming row's, so silent
    /// value-corruption is observable.
    fn push_replace(&mut self, mid: i64, r: CompressionSize, summary: &mut MigrationSummary) {
        if let Some(existing) = self.rows.get(&mid)
            && existing.value_bytes != r.value_bytes
        {
            summary.deduped_with_conflict += 1;
        }
        self.rows.insert(mid, r);
    }

    /// file-sizes-*.json.gz path: per-file rows aggregate into one
    /// `(commit, dataset, dataset_variant, format)` row by summing,
    /// mirroring the prior `value_bytes = compression_sizes.value_bytes
    /// + excluded.value_bytes`.
    fn push_sum(&mut self, mid: i64, r: CompressionSize) {
        let add = r.value_bytes;
        self.rows
            .entry(mid)
            .and_modify(|x| x.value_bytes += add)
            .or_insert(r);
    }
}

fn build_query_batch(a: QueryAccum) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("dataset_variant", DataType::Utf8, true),
        Field::new("scale_factor", DataType::Utf8, true),
        Field::new("query_idx", DataType::Int32, false),
        Field::new("storage", DataType::Utf8, false),
        Field::new("engine", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("value_ns", DataType::Int64, false),
        Field::new(
            "all_runtimes_ns",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        ),
        Field::new("peak_physical", DataType::Int64, true),
        Field::new("peak_virtual", DataType::Int64, true),
        Field::new("physical_delta", DataType::Int64, true),
        Field::new("virtual_delta", DataType::Int64, true),
        Field::new("env_triple", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(a.measurement_id)),
        Arc::new(StringArray::from(a.commit_sha)),
        Arc::new(StringArray::from(a.dataset)),
        Arc::new(StringArray::from(a.dataset_variant)),
        Arc::new(StringArray::from(a.scale_factor)),
        Arc::new(Int32Array::from(a.query_idx)),
        Arc::new(StringArray::from(a.storage)),
        Arc::new(StringArray::from(a.engine)),
        Arc::new(StringArray::from(a.format)),
        Arc::new(Int64Array::from(a.value_ns)),
        Arc::new(build_list_int64(a.all_runtimes_ns)),
        Arc::new(Int64Array::from(a.peak_physical)),
        Arc::new(Int64Array::from(a.peak_virtual)),
        Arc::new(Int64Array::from(a.physical_delta)),
        Arc::new(Int64Array::from(a.virtual_delta)),
        Arc::new(StringArray::from(a.env_triple)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

fn build_compression_time_batch(a: CompressionTimeAccum) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("dataset_variant", DataType::Utf8, true),
        Field::new("format", DataType::Utf8, false),
        Field::new("op", DataType::Utf8, false),
        Field::new("value_ns", DataType::Int64, false),
        Field::new(
            "all_runtimes_ns",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        ),
        Field::new("env_triple", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(a.measurement_id)),
        Arc::new(StringArray::from(a.commit_sha)),
        Arc::new(StringArray::from(a.dataset)),
        Arc::new(StringArray::from(a.dataset_variant)),
        Arc::new(StringArray::from(a.format)),
        Arc::new(StringArray::from(a.op)),
        Arc::new(Int64Array::from(a.value_ns)),
        Arc::new(build_list_int64(a.all_runtimes_ns)),
        Arc::new(StringArray::from(a.env_triple)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

fn build_random_access_batch(a: RandomAccessAccum) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("value_ns", DataType::Int64, false),
        Field::new(
            "all_runtimes_ns",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        ),
        Field::new("env_triple", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(a.measurement_id)),
        Arc::new(StringArray::from(a.commit_sha)),
        Arc::new(StringArray::from(a.dataset)),
        Arc::new(StringArray::from(a.format)),
        Arc::new(Int64Array::from(a.value_ns)),
        Arc::new(build_list_int64(a.all_runtimes_ns)),
        Arc::new(StringArray::from(a.env_triple)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

fn build_compression_size_batch(a: CompressionSizeAccum) -> Result<RecordBatch> {
    let n = a.rows.len();
    let mut measurement_id = Vec::with_capacity(n);
    let mut commit_sha = Vec::with_capacity(n);
    let mut dataset = Vec::with_capacity(n);
    let mut dataset_variant = Vec::with_capacity(n);
    let mut format = Vec::with_capacity(n);
    let mut value_bytes = Vec::with_capacity(n);
    for (mid, cs) in a.rows {
        measurement_id.push(mid);
        commit_sha.push(cs.commit_sha);
        dataset.push(cs.dataset);
        dataset_variant.push(cs.dataset_variant);
        format.push(cs.format);
        value_bytes.push(cs.value_bytes);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("dataset_variant", DataType::Utf8, true),
        Field::new("format", DataType::Utf8, false),
        Field::new("value_bytes", DataType::Int64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(measurement_id)),
        Arc::new(StringArray::from(commit_sha)),
        Arc::new(StringArray::from(dataset)),
        Arc::new(StringArray::from(dataset_variant)),
        Arc::new(StringArray::from(format)),
        Arc::new(Int64Array::from(value_bytes)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

/// Build a non-nullable `List<Int64>` Arrow array from one inner Vec
/// per row. The outer list is non-null; inner i64 values are non-null.
fn build_list_int64(values: Vec<Vec<i64>>) -> ListArray {
    let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
    offsets.push(0);
    let mut flat: Vec<i64> = Vec::new();
    for inner in values {
        flat.extend_from_slice(&inner);
        offsets.push(flat.len() as i32);
    }
    let values_arr = Int64Array::from(flat);
    let field = Arc::new(Field::new("item", DataType::Int64, false));
    ListArray::new(
        field,
        OffsetBuffer::new(offsets.into()),
        Arc::new(values_arr),
        None,
    )
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
