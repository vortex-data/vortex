// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! EXPERIMENT: read-path knobs and counters for the CI A/B. Not for merge.
//!
//! `XP_MODE` selects one variant per process. The counters total the read path's work so the
//! bench can report it per take.

#![allow(missing_docs, clippy::disallowed_methods)]

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// The variant selected by `XP_MODE`.
#[derive(Debug, Default)]
pub struct Knobs {
    pub name: String,
    /// Step the scan's root run on the caller's poll, as on develop.
    pub dev: bool,
    /// Override the local file's read concurrency.
    pub conc: Option<usize>,
    /// Override the local file's coalescing: `Some(None)` turns it off.
    pub coal: Option<Option<(u64, u64)>>,
    /// Read a submitted batch on blocking tasks of up to this many reads each (0: one task).
    pub batch: Option<usize>,
    /// Cap the chunks a scan runs at once.
    pub inflight: Option<usize>,
    /// Hand the read driver only as many reads as it has free slots, and coalesce a read only
    /// when more requests wait than there are slots left.
    pub adapt: bool,
}

pub fn knobs() -> &'static Knobs {
    static KNOBS: OnceLock<Knobs> = OnceLock::new();
    KNOBS.get_or_init(|| {
        let name = std::env::var("XP_MODE").unwrap_or_else(|_| "base".to_string());
        let mut k = Knobs {
            name: name.clone(),
            ..Knobs::default()
        };
        for part in name.split('+') {
            let num = |p: &str| p.parse::<u64>().ok();
            if part == "dev" {
                k.dev = true;
            } else if part == "adapt" {
                k.adapt = true;
            } else if let Some(n) = part.strip_prefix("conc").and_then(num) {
                k.conc = Some(n as usize);
            } else if part == "nocoal" {
                k.coal = Some(None);
            } else if let Some(n) = part.strip_prefix("coal").and_then(num) {
                // coal<N>: N KiB distance, 4 * N KiB max.
                k.coal = Some(Some((n << 10, (4 * n) << 10)));
            } else if part == "batch" {
                k.batch = Some(0);
            } else if let Some(n) = part.strip_prefix("batch").and_then(num) {
                k.batch = Some(n as usize);
            } else if let Some(n) = part.strip_prefix("inflight").and_then(num) {
                k.inflight = Some(n as usize);
            }
        }
        k
    })
}

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
        const ALL: &[(&str, &AtomicU64)] = &[$((stringify!($name), &$name)),*];
    };
}

counters!(
    // Physical reads submitted by the segment source's driver.
    READS,
    // Bytes those reads span.
    READ_BYTES,
    // Segment requests they serve, and the bytes those requests asked for.
    SEGMENTS,
    SEGMENT_BYTES,
    // Time reads waited in the driver for a free concurrency slot.
    QUEUE_NS,
    // Blocking tasks started, the time from submission to start, in the read, and back.
    TASKS,
    SCHED_NS,
    PREAD_NS,
    RETURN_NS,
    // Largest number of reads in flight at once.
    MAX_ACTIVE,
);

pub fn add(counter: &AtomicU64, value: u64) {
    counter.fetch_add(value, Ordering::Relaxed);
}

pub fn max(counter: &AtomicU64, value: u64) {
    counter.fetch_max(value, Ordering::Relaxed);
}

pub fn since(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

pub fn reset() {
    for (_, c) in ALL {
        c.store(0, Ordering::Relaxed);
    }
}

/// The counters per take, as `key=value` pairs.
pub fn summary(takes: usize) -> String {
    let takes = takes.max(1) as f64;
    let get = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64;
    let reads = get(&READS).max(1.0);
    let tasks = get(&TASKS).max(1.0);
    format!(
        "reads={:.1} segs={:.1} read_kib={:.1} seg_kib={:.1} kib_per_read={:.1} queue_us={:.1} \
         tasks={:.1} sched_us={:.2} pread_us={:.2} ret_us={:.2} queue_per_read_us={:.2} max_active={}",
        get(&READS) / takes,
        get(&SEGMENTS) / takes,
        get(&READ_BYTES) / takes / 1024.0,
        get(&SEGMENT_BYTES) / takes / 1024.0,
        get(&READ_BYTES) / reads / 1024.0,
        get(&QUEUE_NS) / takes / 1e3,
        get(&TASKS) / takes,
        get(&SCHED_NS) / tasks / 1e3,
        get(&PREAD_NS) / tasks / 1e3,
        get(&RETURN_NS) / tasks / 1e3,
        get(&QUEUE_NS) / reads / 1e3,
        get(&MAX_ACTIVE) as u64,
    )
}
