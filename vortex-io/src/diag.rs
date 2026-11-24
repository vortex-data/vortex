// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diagnostic work counters, exact rather than timed. Not for merging.
//!
//! Every counter is off until [`enable`] is called, so a disabled counter costs one relaxed load.
//! An enabled counter adds to one of [`SHARDS`] cache-line-padded atomics, picked by thread, so
//! that counting from many threads does not serialize them on one cache line.

use std::cell::Cell;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

const SHARDS: usize = 128;

static ENABLED: AtomicBool = AtomicBool::new(false);
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Starts counting.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Whether counting is on.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[repr(align(128))]
struct Padded(AtomicU64);

/// A counter summed over per-thread shards.
pub struct Counter {
    name: &'static str,
    shards: [Padded; SHARDS],
}

impl Counter {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            shards: [const { Padded(AtomicU64::new(0)) }; SHARDS],
        }
    }

    /// Adds `n`, if counting is on.
    #[inline]
    pub fn add(&self, n: u64) {
        if enabled() {
            self.shards[shard()].0.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Adds one, if counting is on.
    #[inline]
    pub fn inc(&self) {
        self.add(1);
    }

    /// The counter's name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the total and resets it.
    pub fn take(&self) -> u64 {
        self.shards
            .iter()
            .map(|shard| shard.0.swap(0, Ordering::Relaxed))
            .sum()
    }
}

/// The calling thread's shard. It allocates nothing, so a global allocator can call it.
#[inline]
fn shard() -> usize {
    SHARD
        .try_with(|shard| {
            let mut index = shard.get();
            if index == usize::MAX {
                index = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS;
                shard.set(index);
            }
            index
        })
        .unwrap_or(0)
}

/// Entries into a blocking runtime's `block_on`, or into its blocking iterator's `next`.
pub static BLOCK_ON: Counter = Counter::new("block_on");
/// Calls to `Executor::run`, each of which registers a runner with the executor.
pub static RUNNERS: Counter = Counter::new("runners");
/// Planner runs created, `Run::new`.
pub static RUNS: Counter = Counter::new("runs");
/// Clones of a `ScanStream`.
pub static STREAM_CLONES: Counter = Counter::new("stream_clones");
/// Units of work a host scan joined: a split taken, or a stream clone handed out.
pub static JOINS: Counter = Counter::new("joins");
/// Arrays a host scan handed to its engine, including empty ones.
pub static BATCHES: Counter = Counter::new("batches");
/// Empty arrays among [`BATCHES`].
pub static EMPTY_BATCHES: Counter = Counter::new("empty_batches");
/// Heap allocations, including reallocations, made by a counting global allocator.
pub static ALLOCS: Counter = Counter::new("allocs");
/// Bytes requested by those allocations.
pub static ALLOC_BYTES: Counter = Counter::new("alloc_bytes");

/// Every counter, in report order.
pub static ALL: [&Counter; 9] = [
    &BLOCK_ON,
    &RUNNERS,
    &RUNS,
    &STREAM_CLONES,
    &JOINS,
    &BATCHES,
    &EMPTY_BATCHES,
    &ALLOCS,
    &ALLOC_BYTES,
];

/// Resets every counter.
pub fn reset() {
    for counter in ALL {
        counter.take();
    }
}

/// Formats every counter's total, and its total per batch, as one line, and resets them.
pub fn report(label: &str) -> String {
    let totals: Vec<(&str, u64)> = ALL.iter().map(|c| (c.name(), c.take())).collect();
    let batches = totals
        .iter()
        .find(|(name, _)| *name == BATCHES.name())
        .map_or(0, |(_, n)| *n);
    let mut line = format!("VXCOUNT {label}");
    for (name, total) in &totals {
        line.push_str(&format!(" {name}={total}"));
    }
    for (name, total) in &totals {
        if *name == BATCHES.name() {
            continue;
        }
        let per_batch = if batches == 0 {
            0.0
        } else {
            *total as f64 / batches as f64
        };
        line.push_str(&format!(" {name}/batch={per_batch:.2}"));
    }
    line
}
