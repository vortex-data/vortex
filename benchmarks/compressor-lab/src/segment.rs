// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Buffered writing of fact rows and their ledger entries.
//!
//! A writer collects the rows of finished tasks and flushes them as one Parquet segment per
//! table, then one ledger segment naming the tasks. The order is the commit protocol: chunks
//! first (uploaded as tasks finish), then facts, then the ledger, so a ledger row only ever points
//! at data that exists. A crash loses at most the unflushed tasks, which the re-run redoes.

use std::time::Duration;
use std::time::Instant;

use rand::RngExt;

use crate::key::TaskKey;
use crate::plan::TaskKind;
use crate::store::Store;
use crate::table::LedgerRow;
use crate::table::Table;

/// Flush after this many tasks.
const FLUSH_TASKS: usize = 2_000;
/// Flush after this long.
const FLUSH_AFTER: Duration = Duration::from_secs(300);

/// A random id for this process, used in segment names and heartbeats.
pub fn new_worker_id() -> String {
    format!("{:016x}", rand::rng().random::<u64>())
}

/// Buffers one table's rows plus the ledger rows for the tasks that made them.
pub struct SegmentWriter<'a, T: Table> {
    store: &'a Store,
    worker_id: String,
    kind: TaskKind,
    dir: String,
    git_commit: String,
    rows: Vec<T>,
    ledger: Vec<LedgerRow>,
    seq: u64,
    since: Instant,
    /// Tasks written by this writer, flushed or not.
    pub written: Vec<TaskKey>,
}

impl<'a, T: Table> SegmentWriter<'a, T> {
    /// A writer for `kind`'s rows into `dir`.
    pub fn new(
        store: &'a Store,
        worker_id: &str,
        kind: TaskKind,
        dir: String,
        git_commit: &str,
    ) -> Self {
        Self {
            store,
            worker_id: worker_id.to_string(),
            kind,
            dir,
            git_commit: git_commit.to_string(),
            rows: Vec::new(),
            ledger: Vec::new(),
            seq: 0,
            since: Instant::now(),
            written: Vec::new(),
        }
    }

    /// Adds a finished task's rows. Returns the hash of the rows, for nondeterminism checks.
    pub fn add(&mut self, key: &TaskKey, rows: Vec<T>, millis: u128) -> anyhow::Result<String> {
        let output_hash = crate::key::bytes_digest(&serde_json::to_vec(&rows)?);
        self.ledger.push(LedgerRow {
            task_key: key.to_string(),
            kind: self.kind.as_str().to_string(),
            output_hash: output_hash.clone(),
            rows: rows.len() as u64,
            millis: u64::try_from(millis).unwrap_or(u64::MAX),
            git_commit: self.git_commit.clone(),
        });
        self.rows.extend(rows);
        self.written.push(key.clone());
        if self.ledger.len() >= FLUSH_TASKS || self.since.elapsed() >= FLUSH_AFTER {
            self.flush()?;
        }
        Ok(output_hash)
    }

    /// Writes the buffered facts segment, then its ledger segment.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        if self.ledger.is_empty() {
            return Ok(());
        }
        let segment = segment_id(&self.worker_id, self.seq);
        self.seq += 1;
        self.store.put_segment(&self.dir, &segment, &self.rows)?;
        self.store
            .put_segment(&crate::store::ledger_dir(self.kind), &segment, &self.ledger)?;
        self.rows.clear();
        self.ledger.clear();
        self.since = Instant::now();
        self.store.put_json(
            &format!("workers/{}.json", self.worker_id),
            &serde_json::json!({
                "worker_id": self.worker_id,
                "kind": self.kind.as_str(),
                "last_segment": segment,
                "at_ms": crate::store::now_ms(),
            }),
        )?;
        Ok(())
    }
}

impl<T: Table> Drop for SegmentWriter<'_, T> {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("flushing {} rows failed: {e:#}", self.kind.as_str());
        }
    }
}

/// A segment id: UTC time, worker id and sequence number, so names sort by time and never collide.
pub fn segment_id(worker_id: &str, seq: u64) -> String {
    let ms = crate::store::now_ms();
    format!("{ms:013}-{worker_id}-{seq:06}")
}

/// Buffers timing rows and writes them as segments of one machine's partition.
pub struct TimingWriter<'a> {
    store: &'a Store,
    worker_id: String,
    dir: String,
    rows: Vec<crate::table::TimingRow>,
    seq: u64,
}

impl<'a> TimingWriter<'a> {
    /// A writer into `machine_id`'s partition.
    pub fn new(store: &'a Store, worker_id: &str, machine_id: &str) -> Self {
        Self {
            store,
            worker_id: worker_id.to_string(),
            dir: crate::store::timings_dir(machine_id),
            rows: Vec::new(),
            seq: 0,
        }
    }

    /// Adds samples, flushing every 200,000 rows.
    pub fn extend(&mut self, rows: Vec<crate::table::TimingRow>) -> anyhow::Result<()> {
        self.rows.extend(rows);
        if self.rows.len() >= 200_000 {
            self.flush()?;
        }
        Ok(())
    }

    /// Writes the buffered samples.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let segment = segment_id(&self.worker_id, self.seq);
        self.seq += 1;
        self.store.put_segment(&self.dir, &segment, &self.rows)?;
        self.rows.clear();
        Ok(())
    }
}

impl Drop for TimingWriter<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("flushing timings failed: {e:#}");
        }
    }
}
