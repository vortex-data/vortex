// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resource usage for scans that the caller owns, and its export to a metrics registry.
//!
//! A caller gives a [`ScanUsage`] to a scan through
//! [`ScanBuilder::with_usage`](crate::scan::scan_builder::ScanBuilder::with_usage). The caller
//! decides the scope: one value for each file, partition, or query. Many scans can share one value.
//!
//! A [`UsageSegmentSource`](crate::segments::UsageSegmentSource) adds the segment reads of a
//! file to the same value, also for reads that do not go through a scan.
//!
//! [`ScanMetrics`] exports the scan events to a [`MetricsRegistry`] for dashboards.

use std::iter::Sum;
use std::ops::Add;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_metrics::Counter;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

/// One kind of scan event. The discriminant is the index into the counter arrays.
#[derive(Clone, Copy)]
enum ScanEvent {
    SplitAttempted,
    SplitPruned,
    SplitEmptyAfterPredicate,
    FileScanAttempted,
    FileScanPruned,
    SegmentRequested,
    BytesRequested,
}

/// The events that [`ScanMetrics`] exports. Segment reads are usage only.
const METRIC_EVENT_COUNT: usize = 5;

const EVENT_COUNT: usize = 7;

/// Resource usage of all scans that share this value.
///
/// Counts increase while tasks run. A snapshot is final only after all tasks of all
/// scans that share this value stop. Cancellation does not reset counts, and split tasks that
/// nobody polls add no split counts.
#[derive(Debug, Default)]
pub struct ScanUsage {
    counts: [AtomicU64; EVENT_COUNT],
}

impl ScanUsage {
    /// Returns the current counts without a lock.
    pub fn snapshot(&self) -> ScanUsageSnapshot {
        let count = |event: ScanEvent| self.counts[event as usize].load(Ordering::Relaxed);

        ScanUsageSnapshot {
            splits_attempted: count(ScanEvent::SplitAttempted),
            splits_pruned: count(ScanEvent::SplitPruned),
            splits_empty_after_predicate: count(ScanEvent::SplitEmptyAfterPredicate),
            file_scans_attempted: count(ScanEvent::FileScanAttempted),
            file_scans_pruned: count(ScanEvent::FileScanPruned),
            segments_requested: count(ScanEvent::SegmentRequested),
            bytes_requested: count(ScanEvent::BytesRequested),
        }
    }

    /// Adds one segment read of `bytes` bytes.
    pub(crate) fn record_segment(&self, bytes: usize) {
        self.add(ScanEvent::SegmentRequested, 1);
        self.add(ScanEvent::BytesRequested, bytes as u64);
    }

    fn add(&self, event: ScanEvent, value: u64) {
        self.counts[event as usize].fetch_add(value, Ordering::Relaxed);
    }
}

/// The counts of a [`ScanUsage`] at one point in time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScanUsageSnapshot {
    /// Split tasks that started execution.
    /// An eager rejection through the layout data source counts its requested range as one split.
    pub splits_attempted: u64,
    /// Splits that split statistics fully rejected.
    pub splits_pruned: u64,
    /// Splits that predicate evaluation fully rejected.
    pub splits_empty_after_predicate: u64,
    /// Scan executions that reach a nonempty selection and a nonzero row budget.
    pub file_scans_attempted: u64,
    /// Scan executions that whole-file statistics rejected, at most one for each execution.
    pub file_scans_pruned: u64,
    /// Segment reads that completed through a usage-tracking segment source.
    /// Reads that a cache serves count. Reads that fail do not count.
    pub segments_requested: u64,
    /// Bytes of the segments that [`Self::segments_requested`] counts.
    /// Gaps between segments that the I/O layer merges into one request do not count.
    pub bytes_requested: u64,
}

impl Add for ScanUsageSnapshot {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            splits_attempted: self.splits_attempted + other.splits_attempted,
            splits_pruned: self.splits_pruned + other.splits_pruned,
            splits_empty_after_predicate: self.splits_empty_after_predicate
                + other.splits_empty_after_predicate,
            file_scans_attempted: self.file_scans_attempted + other.file_scans_attempted,
            file_scans_pruned: self.file_scans_pruned + other.file_scans_pruned,
            segments_requested: self.segments_requested + other.segments_requested,
            bytes_requested: self.bytes_requested + other.bytes_requested,
        }
    }
}

impl Sum for ScanUsageSnapshot {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), Add::add)
    }
}

impl ScanUsageSnapshot {
    /// Returns the counts that accrued after `earlier`, an earlier snapshot of the same usage.
    ///
    /// Counts only increase, so no difference is negative.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            splits_attempted: self.splits_attempted - earlier.splits_attempted,
            splits_pruned: self.splits_pruned - earlier.splits_pruned,
            splits_empty_after_predicate: self.splits_empty_after_predicate
                - earlier.splits_empty_after_predicate,
            file_scans_attempted: self.file_scans_attempted - earlier.file_scans_attempted,
            file_scans_pruned: self.file_scans_pruned - earlier.file_scans_pruned,
            segments_requested: self.segments_requested - earlier.segments_requested,
            bytes_requested: self.bytes_requested - earlier.bytes_requested,
        }
    }
}

/// Registry counters that receive the same events as [`ScanUsage`].
///
/// Construction registers five counters. Clones share their handles and add no registry entries,
/// so reuse one value across prepared scans.
#[derive(Clone)]
pub struct ScanMetrics(Arc<[Counter; METRIC_EVENT_COUNT]>);

impl ScanMetrics {
    /// Registers the scan counters in `registry`.
    pub fn new(registry: &dyn MetricsRegistry) -> Self {
        let counter = |name: &'static str| MetricBuilder::new(registry).counter(name);

        Self(Arc::new([
            counter("scan.splits.attempted"),
            counter("scan.splits.pruned"),
            counter("scan.splits.empty_after_predicate"),
            counter("scan.file_scans.attempted"),
            counter("scan.file_scans.pruned"),
        ]))
    }

    fn add(&self, event: ScanEvent) {
        self.0[event as usize].add(1);
    }
}

/// Records the events of one scan execution into its usage and registry sinks.
///
/// File events count at most once for each execution, also when many splits observe them.
pub(crate) struct ScanRecorder {
    usage: Option<Arc<ScanUsage>>,
    metrics: Option<ScanMetrics>,
    file_scan_attempted: AtomicBool,
    file_scan_pruned: AtomicBool,
}

impl ScanRecorder {
    /// Returns `None` when no sink is set, so that scan tasks can skip all recording.
    pub(crate) fn new(usage: Option<Arc<ScanUsage>>, metrics: Option<ScanMetrics>) -> Option<Self> {
        if usage.is_none() && metrics.is_none() {
            return None;
        }

        Some(Self {
            usage,
            metrics,
            file_scan_attempted: AtomicBool::new(false),
            file_scan_pruned: AtomicBool::new(false),
        })
    }

    pub(crate) fn attempt_split(&self) {
        self.record(ScanEvent::SplitAttempted);
    }

    pub(crate) fn prune_split(&self) {
        self.record(ScanEvent::SplitPruned);
    }

    pub(crate) fn empty_split_after_predicate(&self) {
        self.record(ScanEvent::SplitEmptyAfterPredicate);
    }

    pub(crate) fn attempt_file_scan(&self) {
        if !self.file_scan_attempted.swap(true, Ordering::Relaxed) {
            self.record(ScanEvent::FileScanAttempted);
        }
    }

    pub(crate) fn prune_file(&self) {
        if !self.file_scan_pruned.swap(true, Ordering::Relaxed) {
            self.record(ScanEvent::FileScanPruned);
        }
    }

    fn record(&self, event: ScanEvent) {
        if let Some(usage) = &self.usage {
            usage.add(event, 1);
        }

        if let Some(metrics) = &self.metrics {
            metrics.add(event);
        }
    }
}
