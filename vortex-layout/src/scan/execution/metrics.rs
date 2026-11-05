// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Counters of the work a V2 scan did and the work it avoided.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// Counts what a V2 scan prunes, filters and reads. Each counter is a relaxed atomic add once
/// per chunk, pruning plan, conjunct evaluation or segment request, never per row, so a scan
/// always counts.
///
/// An engine hands a scan its counters with
/// [`ScanBuilder::with_metrics`](crate::scan::v2::ScanBuilder::with_metrics) and reads them with
/// [`take`](Self::take) as the scan runs or once it ends.
#[derive(Debug, Default)]
pub struct ScanMetrics {
    chunks: AtomicU64,
    chunks_pruned: AtomicU64,
    rows_pruned_static: AtomicU64,
    rows_pruned_dynamic: AtomicU64,
    rows_read: AtomicU64,
    rows_filtered_static: AtomicU64,
    rows_filtered_dynamic: AtomicU64,
    segments_read: AtomicU64,
    bytes_read: AtomicU64,
    segment_cache_hits: AtomicU64,
}

/// The values of [`ScanMetrics`] at one point of a scan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanMetricsSnapshot {
    /// Chunks the scan cut its rows into.
    pub chunks: u64,
    /// Chunks never planned because the selection or zone pruning left none of their rows.
    pub chunks_pruned: u64,
    /// Rows zone maps excluded for a conjunct without dynamic expressions.
    pub rows_pruned_static: u64,
    /// Rows zone maps excluded for a conjunct with dynamic expressions, such as a join's keys.
    pub rows_pruned_dynamic: u64,
    /// Selected rows of the chunks the scan planned, before any conjunct ran on them.
    pub rows_read: u64,
    /// Rows a conjunct without dynamic expressions rejected.
    pub rows_filtered_static: u64,
    /// Rows a conjunct with dynamic expressions rejected.
    pub rows_filtered_dynamic: u64,
    /// Segments read from the file's segment source.
    pub segments_read: u64,
    /// Bytes of the segments read from the file's segment source.
    pub bytes_read: u64,
    /// Segment requests served from the segments the file's scans retain.
    pub segment_cache_hits: u64,
}

impl ScanMetrics {
    /// Returns the counts since the last take, and resets them, so an engine that takes them
    /// more than once counts every unit of work once.
    pub fn take(&self) -> ScanMetricsSnapshot {
        let take = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed);
        ScanMetricsSnapshot {
            chunks: take(&self.chunks),
            chunks_pruned: take(&self.chunks_pruned),
            rows_pruned_static: take(&self.rows_pruned_static),
            rows_pruned_dynamic: take(&self.rows_pruned_dynamic),
            rows_read: take(&self.rows_read),
            rows_filtered_static: take(&self.rows_filtered_static),
            rows_filtered_dynamic: take(&self.rows_filtered_dynamic),
            segments_read: take(&self.segments_read),
            bytes_read: take(&self.bytes_read),
            segment_cache_hits: take(&self.segment_cache_hits),
        }
    }

    pub(crate) fn add_chunk(&self, planned: bool) {
        self.chunks.fetch_add(1, Ordering::Relaxed);
        if !planned {
            self.chunks_pruned.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn add_rows_pruned(&self, dynamic: bool, rows: usize) {
        let counter = if dynamic {
            &self.rows_pruned_dynamic
        } else {
            &self.rows_pruned_static
        };
        counter.fetch_add(rows as u64, Ordering::Relaxed);
    }

    pub(crate) fn add_rows_read(&self, rows: usize) {
        self.rows_read.fetch_add(rows as u64, Ordering::Relaxed);
    }

    pub(crate) fn add_rows_filtered(&self, dynamic: bool, rows: usize) {
        let counter = if dynamic {
            &self.rows_filtered_dynamic
        } else {
            &self.rows_filtered_static
        };
        counter.fetch_add(rows as u64, Ordering::Relaxed);
    }

    pub(crate) fn add_segment_read(&self, bytes: usize) {
        self.segments_read.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn add_segment_cache_hit(&self) {
        self.segment_cache_hits.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::ScanMetrics;
    use super::ScanMetricsSnapshot;

    #[test]
    fn take_returns_counts_since_the_last_take() {
        let metrics = ScanMetrics::default();
        metrics.add_chunk(true);
        metrics.add_chunk(false);
        metrics.add_rows_pruned(false, 10);
        metrics.add_rows_pruned(true, 3);
        metrics.add_rows_read(7);
        metrics.add_rows_filtered(true, 2);
        metrics.add_segment_read(100);
        metrics.add_segment_cache_hit();

        assert_eq!(
            metrics.take(),
            ScanMetricsSnapshot {
                chunks: 2,
                chunks_pruned: 1,
                rows_pruned_static: 10,
                rows_pruned_dynamic: 3,
                rows_read: 7,
                rows_filtered_static: 0,
                rows_filtered_dynamic: 2,
                segments_read: 1,
                bytes_read: 100,
                segment_cache_hits: 1,
            }
        );
        assert_eq!(metrics.take(), ScanMetricsSnapshot::default());
    }
}
