// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Counters for individual scan executions and totals for a prepared scan.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_metrics::Counter;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

/// Observed counts for one scan execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanCounterSnapshot {
    /// Selected split tasks that started execution.
    pub splits_considered: u64,
    /// Splits that split statistics fully rejected.
    pub splits_pruned: u64,
    /// Splits that predicate evaluation fully rejected.
    pub splits_filtered: u64,
    /// Whole-file statistics rejections, at most one per execution.
    pub files_pruned: u64,
}

/// Live counters for one scan execution.
///
/// Cancellation does not reset these counters. Counts can advance until active tasks stop.
/// Unpolled tasks contribute no counts. Snapshots during execution can contain partial counts.
#[derive(Debug, Default)]
pub struct ScanCounters {
    splits_considered: AtomicU64,
    splits_pruned: AtomicU64,
    splits_filtered: AtomicU64,
    files_pruned: AtomicU64,
}

impl ScanCounters {
    /// Returns the current counts without a lock.
    ///
    /// The snapshot is consistent after all split tasks stop.
    pub fn snapshot(&self) -> ScanCounterSnapshot {
        ScanCounterSnapshot {
            splits_considered: self.splits_considered.load(Ordering::Relaxed),
            splits_pruned: self.splits_pruned.load(Ordering::Relaxed),
            splits_filtered: self.splits_filtered.load(Ordering::Relaxed),
            files_pruned: self.files_pruned.load(Ordering::Relaxed),
        }
    }

    pub(super) fn consider_split(&self) {
        self.splits_considered.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn prune_split(&self) {
        self.splits_pruned.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn filter_split(&self) {
        self.splits_filtered.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn prune_file(&self) {
        if self.files_pruned.load(Ordering::Relaxed) == 0 {
            self.files_pruned.store(1, Ordering::Relaxed);
        }
    }
}

#[derive(Clone)]
pub(super) struct ScanMetrics {
    splits_considered: Counter,
    splits_pruned: Counter,
    splits_filtered: Counter,
    files_pruned: Counter,
}

impl ScanMetrics {
    pub fn new(registry: &dyn MetricsRegistry) -> Self {
        Self {
            splits_considered: MetricBuilder::new(registry).counter("scan.splits.considered"),
            splits_pruned: MetricBuilder::new(registry).counter("scan.splits.pruned"),
            splits_filtered: MetricBuilder::new(registry).counter("scan.splits.filtered"),
            files_pruned: MetricBuilder::new(registry).counter("scan.files.pruned"),
        }
    }

    pub fn record(&self, snapshot: ScanCounterSnapshot) {
        for (counter, value) in [
            (&self.splits_considered, snapshot.splits_considered),
            (&self.splits_pruned, snapshot.splits_pruned),
            (&self.splits_filtered, snapshot.splits_filtered),
            (&self.files_pruned, snapshot.files_pruned),
        ] {
            if value != 0 {
                counter.add(value);
            }
        }
    }
}
