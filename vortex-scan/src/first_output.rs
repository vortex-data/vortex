// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! When a process's scans last handed a batch to their engine for the first time.
//!
//! Engines report through different APIs, so a benchmark that compares them measures time to
//! first row here: it calls [`reset`] as a query starts, the engine integrations call [`record`]
//! for each batch, and [`since`] gives the delay.

use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

const NONE: u64 = u64::MAX;

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);
static FIRST: AtomicU64 = AtomicU64::new(NONE);

/// Forgets the batch recorded so far.
pub fn reset() {
    LazyLock::force(&EPOCH);
    FIRST.store(NONE, Ordering::Relaxed);
}

/// Records that a scan handed a batch to its engine.
#[inline]
pub fn record() {
    // Every batch after the first of a query stops at this load.
    if FIRST.load(Ordering::Relaxed) == NONE {
        let nanos = u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(NONE - 1);
        FIRST.fetch_min(nanos, Ordering::Relaxed);
    }
}

/// The delay from `start` to the first batch since [`reset`], if there was one.
pub fn since(start: Instant) -> Option<Duration> {
    let nanos = FIRST.load(Ordering::Relaxed);
    (nanos != NONE).then(|| (*EPOCH + Duration::from_nanos(nanos)).saturating_duration_since(start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_first_batch_until_reset() {
        let start = Instant::now();
        reset();
        assert_eq!(since(start), None);
        record();
        let first = since(start);
        assert!(first.is_some());
        record();
        assert_eq!(since(start), first);
        reset();
        assert_eq!(since(start), None);
    }
}
