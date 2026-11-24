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

/// The rows one fact task writes, possibly into several tables.
pub trait Batch: Default + serde::Serialize {
    /// Appends another task's rows.
    fn append(&mut self, other: Self);
    /// Writes the rows as segment `segment` of each table.
    fn write(&self, store: &Store, segment: &str) -> anyhow::Result<()>;
    /// Drops the buffered rows.
    fn clear(&mut self);
    /// How many rows are buffered.
    fn len(&self) -> usize;
    /// Whether no rows are buffered.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The rows of one table, at one version.
#[derive(Debug, serde::Serialize)]
pub struct Rows<T: Table> {
    /// The table version.
    #[serde(skip)]
    pub version: u32,
    /// The rows.
    pub rows: Vec<T>,
}

impl<T: Table> Default for Rows<T> {
    fn default() -> Self {
        Self {
            version: 0,
            rows: Vec::new(),
        }
    }
}

impl<T: Table> Rows<T> {
    /// Rows of version `version`.
    pub fn new(version: u32, rows: Vec<T>) -> Self {
        Self { version, rows }
    }
}

impl<T: Table> Batch for Rows<T> {
    fn append(&mut self, other: Self) {
        self.version = other.version;
        self.rows.extend(other.rows);
    }

    fn write(&self, store: &Store, segment: &str) -> anyhow::Result<()> {
        store.put_segment(&T::dir(self.version), segment, &self.rows)?;
        Ok(())
    }

    fn clear(&mut self) {
        self.rows.clear();
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}

/// Buffers fact rows plus the ledger rows for the tasks that made them.
pub struct SegmentWriter<'a, B: Batch> {
    store: &'a Store,
    worker_id: String,
    kind: TaskKind,
    git_commit: String,
    rows: B,
    ledger: Vec<LedgerRow>,
    seq: u64,
    since: Instant,
}

impl<'a, B: Batch> SegmentWriter<'a, B> {
    /// A writer for `kind`'s rows.
    pub fn new(store: &'a Store, worker_id: &str, kind: TaskKind, git_commit: &str) -> Self {
        Self {
            store,
            worker_id: worker_id.to_string(),
            kind,
            git_commit: git_commit.to_string(),
            rows: B::default(),
            ledger: Vec::new(),
            seq: 0,
            since: Instant::now(),
        }
    }

    /// Adds a finished task's rows. Returns the hash of the rows, for nondeterminism checks.
    pub fn add(&mut self, key: &TaskKey, rows: B, millis: u128) -> anyhow::Result<String> {
        let output_hash = crate::key::bytes_digest(&serde_json::to_vec(&rows)?);
        self.ledger.push(LedgerRow {
            task_key: key.to_string(),
            kind: self.kind.as_str().to_string(),
            output_hash: output_hash.clone(),
            rows: rows.len() as u64,
            millis: u64::try_from(millis).unwrap_or(u64::MAX),
            git_commit: self.git_commit.clone(),
        });
        self.rows.append(rows);
        if self.ledger.len() >= FLUSH_TASKS || self.since.elapsed() >= FLUSH_AFTER {
            self.flush()?;
        }
        Ok(output_hash)
    }

    /// Writes the buffered facts segments, then their ledger segment.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        if self.ledger.is_empty() {
            return Ok(());
        }
        let segment = segment_id(&self.worker_id, self.seq);
        self.seq += 1;
        self.rows.write(self.store, &segment)?;
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

impl<B: Batch> Drop for SegmentWriter<'_, B> {
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

/// Buffers rows of a per-machine table and writes them as segments of one machine's partition.
pub struct MachineWriter<'a, T: Table> {
    store: &'a Store,
    worker_id: String,
    dir: String,
    rows: Vec<T>,
    seq: u64,
}

/// Writes timing samples.
pub type TimingWriter<'a> = MachineWriter<'a, crate::table::TimingRow>;

impl<'a, T: Table> MachineWriter<'a, T> {
    /// A writer into `machine_id`'s partition.
    pub fn new(store: &'a Store, worker_id: &str, machine_id: &str) -> Self {
        Self {
            store,
            worker_id: worker_id.to_string(),
            dir: crate::store::machine_dir::<T>(machine_id),
            rows: Vec::new(),
            seq: 0,
        }
    }

    /// Adds rows, flushing every 200,000.
    pub fn extend(&mut self, rows: impl IntoIterator<Item = T>) -> anyhow::Result<()> {
        self.rows.extend(rows);
        if self.rows.len() >= 200_000 {
            self.flush()?;
        }
        Ok(())
    }

    /// Writes the buffered rows.
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

impl<T: Table> Drop for MachineWriter<'_, T> {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("flushing {} failed: {e:#}", T::NAME);
        }
    }
}
