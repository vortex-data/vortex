// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Store maintenance: compacting small segments and collecting garbage.
//!
//! Compaction writes the merged segment first and deletes its inputs second. A reader that lists
//! in between sees both and drops the duplicates, so compaction is safe while workers append.
//!
//! GC keeps every raw file under `raw/` that a live plan (one without a `RETIRED` marker) still
//! reads, and removes orphan fact segments (written without their ledger segment, by a crashed
//! worker). It only runs when no worker has written a heartbeat recently: a worker that finds a raw
//! file already stored skips the copy, so a file GC judged dead could otherwise be reused under it.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use anyhow::bail;
use serde::Serialize;

use crate::chunks::CHUNKS_VERSION;
use crate::plan::CANDIDATES_VERSION;
use crate::plan::FEATURES_VERSION;
use crate::plan::Task;
use crate::plan::TaskInput;
use crate::plan::TaskKind;
use crate::source::ChunkInput;
use crate::store::Listed;
use crate::store::Store;
use crate::table::ChunkRow;
use crate::table::EncodingRow;
use crate::table::FeatureRow;
use crate::table::LedgerRow;
use crate::table::SearchRow;
use crate::table::SiteFactRow;
use crate::table::Table;
use crate::table::TimingRow;

/// Objects younger than this are never collected: a worker may still be committing them.
const GRACE_MS: i64 = 48 * 3600 * 1000;
/// A heartbeat younger than this means a worker may be running.
const ACTIVE_MS: i64 = 15 * 60 * 1000;
/// A compaction lock older than this is taken over.
const LOCK_MS: i64 = 3600 * 1000;

/// What maintenance did, or would do on a dry run.
#[derive(Debug, Default, Serialize)]
pub struct Report {
    /// Segments merged away.
    pub segments_merged: u64,
    /// Segments written by compaction.
    pub segments_written: u64,
    /// Raw files deleted.
    pub raw_deleted: u64,
    /// Bytes of raw files deleted.
    pub raw_bytes_deleted: u64,
    /// Orphan fact segments deleted.
    pub orphans_deleted: u64,
    /// Raw files kept because a live plan reads them.
    pub raw_live: u64,
    /// Whether anything was actually deleted.
    pub dry_run: bool,
}

/// Records a plan's task list in the store, so GC can tell which raw files it reads.
pub fn record_tasks(store: &Store, plan_id: &str, tasks: &[Task]) -> anyhow::Result<()> {
    let key = format!("plans/{plan_id}/tasks.jsonl");
    if store.get(&key)?.is_some() {
        return Ok(());
    }
    let mut out = Vec::new();
    for task in tasks {
        serde_json::to_writer(&mut out, task)?;
        out.push(b'\n');
    }
    store.put_create(&key, out)?;
    Ok(())
}

fn compact_dir<T: Table>(
    store: &Store,
    dir: &str,
    segment: &str,
    dry_run: bool,
    report: &mut Report,
) -> anyhow::Result<()> {
    let segments: Vec<Listed> = store
        .list(dir)?
        .into_iter()
        .filter(|l| l.key.ends_with(".parquet"))
        .collect();
    if segments.len() < 2 {
        return Ok(());
    }
    report.segments_merged += segments.len() as u64;
    report.segments_written += 1;
    if dry_run {
        return Ok(());
    }
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for segment in &segments {
        for row in store.read_segment::<T>(&segment.key)? {
            if seen.insert(row.dedupe_key()) {
                rows.push(row);
            }
        }
    }
    store.put_segment(dir, segment, &rows)?;
    for old in &segments {
        store.delete(&old.key)?;
    }
    Ok(())
}

/// Merges every table's segments into one per directory.
///
/// Fact and ledger segments are compacted under the same new name, so a compacted fact segment
/// keeps a ledger segment and GC never takes it for an orphan.
pub fn compact(store: &Store, dry_run: bool) -> anyhow::Result<Report> {
    let lock = "locks/compact.json";
    let worker_id = crate::segment::new_worker_id();
    let now = crate::store::now_ms();
    if !dry_run {
        let body = serde_json::to_vec(&serde_json::json!({"worker_id": worker_id, "at_ms": now}))?;
        if !store.put_create(lock, body.clone())? {
            let stale = store
                .list(lock)?
                .first()
                .is_some_and(|l| now - l.modified_ms > LOCK_MS);
            if !stale {
                bail!("another compaction holds {lock}; delete it if that process is gone");
            }
            store.put(lock, body)?;
        }
    }
    let mut report = Report {
        dry_run,
        ..Report::default()
    };
    let segment = crate::segment::segment_id(&format!("compact-{worker_id}"), 0);
    let result = (|| {
        compact_dir::<ChunkRow>(
            store,
            &ChunkRow::dir(CHUNKS_VERSION),
            &segment,
            dry_run,
            &mut report,
        )?;
        compact_dir::<FeatureRow>(
            store,
            &FeatureRow::dir(FEATURES_VERSION),
            &segment,
            dry_run,
            &mut report,
        )?;
        compact_dir::<EncodingRow>(
            store,
            &EncodingRow::dir(CANDIDATES_VERSION),
            &segment,
            dry_run,
            &mut report,
        )?;
        compact_dir::<SiteFactRow>(
            store,
            &SiteFactRow::dir(CANDIDATES_VERSION),
            &segment,
            dry_run,
            &mut report,
        )?;
        compact_dir::<SearchRow>(
            store,
            &SearchRow::dir(CANDIDATES_VERSION),
            &segment,
            dry_run,
            &mut report,
        )?;
        for kind in [TaskKind::Chunk, TaskKind::Features, TaskKind::Candidates] {
            compact_dir::<LedgerRow>(
                store,
                &crate::store::ledger_dir(kind),
                &segment,
                dry_run,
                &mut report,
            )?;
        }
        for machine in store.timing_machines()? {
            compact_dir::<TimingRow>(
                store,
                &crate::store::timings_dir(&machine),
                &segment,
                dry_run,
                &mut report,
            )?;
        }
        anyhow::Ok(())
    })();
    if !dry_run {
        store.delete(lock)?;
    }
    result?;
    Ok(report)
}

/// The SHA-256 of every raw file a live plan reads.
fn live_raw(store: &Store) -> anyhow::Result<BTreeSet<String>> {
    let mut plans: BTreeMap<String, (bool, bool)> = BTreeMap::new();
    for listed in store.list("plans")? {
        let mut parts = listed.key.split('/');
        let (Some(_), Some(id), Some(file)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let entry = plans.entry(id.to_string()).or_default();
        match file {
            "tasks.jsonl" => entry.0 = true,
            "RETIRED" => entry.1 = true,
            _ => {}
        }
    }
    let mut live = BTreeSet::new();
    for (id, (has_tasks, retired)) in plans {
        if retired {
            continue;
        }
        if !has_tasks {
            bail!("plan {id} has no tasks.jsonl in the store, so GC cannot tell what it uses");
        }
        let bytes = store
            .get(&format!("plans/{id}/tasks.jsonl"))?
            .unwrap_or_default();
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let task: Task = serde_json::from_slice(line)?;
            if let TaskInput::Chunk(
                ChunkInput::Parquet {
                    file_fingerprint, ..
                }
                | ChunkInput::Remote {
                    file_fingerprint, ..
                },
            ) = task.input
            {
                live.insert(file_fingerprint);
            }
        }
    }
    Ok(live)
}

/// Collects raw files no live plan reads, and orphan fact segments.
pub fn collect(store: &Store, dry_run: bool, force: bool) -> anyhow::Result<Report> {
    let now = crate::store::now_ms();
    let active: Vec<String> = store
        .list("workers")?
        .into_iter()
        .filter(|l| now - l.modified_ms < ACTIVE_MS)
        .map(|l| l.key)
        .collect();
    if !active.is_empty() && !force {
        bail!(
            "{} worker(s) wrote a heartbeat in the last 15 minutes ({}); run GC between runs, or \
             pass --force if they are gone",
            active.len(),
            active.join(", ")
        );
    }
    let mut report = Report {
        dry_run,
        ..Report::default()
    };

    let live = live_raw(store)?;
    for listed in store.list("raw")? {
        let hash = listed.key.rsplit('/').next().unwrap_or_default();
        if live.contains(hash) {
            report.raw_live += 1;
            continue;
        }
        if now - listed.modified_ms < GRACE_MS {
            continue;
        }
        report.raw_deleted += 1;
        report.raw_bytes_deleted += listed.size;
        if !dry_run {
            store.delete(&listed.key)?;
        }
    }

    let committed: BTreeSet<String> = store
        .list("ledger")?
        .iter()
        .filter_map(|l| segment_name(&l.key))
        .collect();
    for listed in store.list("facts")? {
        let Some(name) = segment_name(&listed.key) else {
            continue;
        };
        // Compaction merges facts and ledger separately, so its segments are never orphans.
        if committed.contains(&name)
            || name.contains("-compact-")
            || now - listed.modified_ms < GRACE_MS
        {
            continue;
        }
        report.orphans_deleted += 1;
        if !dry_run {
            store.delete(&listed.key)?;
        }
    }
    Ok(report)
}

fn segment_name(key: &str) -> Option<String> {
    key.rsplit('/')
        .next()?
        .strip_suffix(".parquet")
        .map(str::to_string)
}
