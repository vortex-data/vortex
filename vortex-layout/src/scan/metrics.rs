// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Counters for individual scan executions and shared aggregate totals.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_metrics::Counter;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

/// Observed counts for one scan execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanCounterSnapshot {
    /// Selected split tasks that started execution.
    /// An eager rejection through the layout data source counts its requested range as one split.
    pub splits_attempted: u64,
    /// Splits that split statistics fully rejected.
    pub splits_pruned: u64,
    /// Splits that predicate evaluation fully rejected.
    pub splits_empty_after_predicate: u64,
    /// File scans that reach a nonempty selection and a nonzero row budget.
    /// This includes scans that whole-file statistics reject before any split starts.
    pub file_scans_attempted: u64,
    /// Whole-file statistics rejections, at most one per execution.
    pub file_scans_pruned: u64,
}

/// Live counters for one scan execution.
///
/// Cancellation does not reset these counters. Counts can advance until active tasks stop.
/// Unpolled split tasks contribute no split counts. File checks can count before split tasks start.
/// Snapshots during execution can contain partial counts.
#[derive(Debug, Default)]
pub struct ScanCounters {
    splits_attempted: AtomicU64,
    splits_pruned: AtomicU64,
    splits_empty_after_predicate: AtomicU64,
    file_scans_attempted: AtomicU64,
    file_scans_pruned: AtomicU64,
}

impl ScanCounters {
    /// Returns the current counts without a lock.
    ///
    /// The snapshot is consistent after all split tasks stop.
    pub fn snapshot(&self) -> ScanCounterSnapshot {
        ScanCounterSnapshot {
            splits_attempted: self.splits_attempted.load(Ordering::Relaxed),
            splits_pruned: self.splits_pruned.load(Ordering::Relaxed),
            splits_empty_after_predicate: self.splits_empty_after_predicate.load(Ordering::Relaxed),
            file_scans_attempted: self.file_scans_attempted.load(Ordering::Relaxed),
            file_scans_pruned: self.file_scans_pruned.load(Ordering::Relaxed),
        }
    }

    pub(super) fn attempt_split(&self) {
        self.splits_attempted.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn prune_split(&self) {
        self.splits_pruned.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn empty_split_after_predicate(&self) {
        self.splits_empty_after_predicate
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn attempt_file_scan(&self) {
        self.file_scans_attempted.store(1, Ordering::Relaxed);
    }

    pub(super) fn prune_file(&self) {
        if self.file_scans_pruned.load(Ordering::Relaxed) == 0 {
            self.file_scans_pruned.store(1, Ordering::Relaxed);
        }
    }
}

/// Aggregate counters shared by scans within one data source or query.
///
/// Construction registers five counters. Clones reuse their handles without new registry entries.
/// Each execution adds its local counts after all of its tasks stop.
/// File scan counts describe executions, not unique physical files or guaranteed I/O savings.
/// These counters exclude engine-level file rejection before a native scan exists.
#[derive(Clone)]
pub struct ScanMetrics(Arc<ScanMetricHandles>);

struct ScanMetricHandles {
    splits_attempted: Counter,
    splits_pruned: Counter,
    splits_empty_after_predicate: Counter,
    file_scans_attempted: Counter,
    file_scans_pruned: Counter,
}

impl ScanMetrics {
    /// Registers aggregate counters in `registry`.
    ///
    /// Reuse this value across prepared scans to keep the number of scan metrics constant.
    pub fn new(registry: &dyn MetricsRegistry) -> Self {
        Self(Arc::new(ScanMetricHandles {
            splits_attempted: MetricBuilder::new(registry).counter("scan.splits.attempted"),
            splits_pruned: MetricBuilder::new(registry).counter("scan.splits.pruned"),
            splits_empty_after_predicate: MetricBuilder::new(registry)
                .counter("scan.splits.empty_after_predicate"),
            file_scans_attempted: MetricBuilder::new(registry).counter("scan.file_scans.attempted"),
            file_scans_pruned: MetricBuilder::new(registry).counter("scan.file_scans.pruned"),
        }))
    }

    pub(super) fn record(&self, snapshot: ScanCounterSnapshot) {
        let handles = &self.0;
        for (counter, value) in [
            (&handles.splits_attempted, snapshot.splits_attempted),
            (&handles.splits_pruned, snapshot.splits_pruned),
            (
                &handles.splits_empty_after_predicate,
                snapshot.splits_empty_after_predicate,
            ),
            (&handles.file_scans_attempted, snapshot.file_scans_attempted),
            (&handles.file_scans_pruned, snapshot.file_scans_pruned),
        ] {
            if value != 0 {
                counter.add(value);
            }
        }
    }
}
