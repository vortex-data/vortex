// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A shared monotonic clock for scan-driver and physical-read diagnostics.

use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Nanoseconds since the first diagnostic event in this process.
///
/// Only call this when the diagnostic tracing target is enabled. All scan and IO workers share
/// the clock, so their intervals can be joined without relying on wall-clock timestamps.
pub fn timestamp_ns() -> u64 {
    static START: LazyLock<Instant> = LazyLock::new(Instant::now);
    u64::try_from(START.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// A process-unique diagnostic identity, never reused when a scan or source is dropped.
pub fn next_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
