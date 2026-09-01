// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-run counters. The eval matrix in the prototype plan records these per row.

use std::time::Duration;

/// Counters accumulated by one driving thread, summed across threads at the end of a run.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    /// Morsels driven.
    pub morsels: u64,
    /// IO uses named by planning streams.
    pub io_uses: u64,
    /// Reads actually issued to the segment source.
    pub io_requests: u64,
    /// Required/speculative scheduler batches containing those reads.
    pub io_batches: u64,
    /// Uses that found a cell already named inside the same morsel.
    pub io_cell_hits: u64,
    /// Uses that went through registration.
    pub io_registered: u64,
    /// Bytes returned by the segment source.
    pub io_bytes: u64,
    /// Number of times a background segment future returned `Pending`.
    pub io_waits: u64,
    /// Inline non-blocking read attempts made by execution.
    pub nowait_attempts: u64,
    /// Inline non-blocking reads satisfied immediately.
    pub nowait_hits: u64,
    /// Inline non-blocking reads that would have waited on storage.
    pub nowait_misses: u64,
    /// Inline non-blocking reads unsupported by the source or filesystem.
    pub nowait_unsupported: u64,
    /// Cumulative wall latency from a segment future's first `Pending` until it became ready.
    ///
    /// Futures overlap and no CPU worker is parked, so this is not additive CPU or scan time.
    pub io_wait_time: Duration,
    /// Segment decodes performed.
    pub decodes: u64,
    /// Decodes served from a shared cell published by another morsel.
    pub decode_reuses: u64,
    /// Conjuncts skipped because the mask was already all-false.
    pub conjuncts_short_circuited: u64,
    /// Morsels whose filter selected no rows.
    pub morsels_empty: u64,
    /// Exact-ticket suspensions returned by execution nodes.
    pub execute_io_blocks: u64,
    /// Morsels that suspended at least once on IO.
    pub morsels_blocked_for_io: u64,
    /// Minimum logical IO uses named by one morsel.
    pub io_uses_per_morsel_min: Option<u64>,
    /// Maximum logical IO uses named by one morsel.
    pub io_uses_per_morsel_max: u64,
    /// Minimum new scan-wide segment requests created by one morsel.
    pub io_requests_per_morsel_min: Option<u64>,
    /// Maximum new scan-wide segment requests created by one morsel.
    pub io_requests_per_morsel_max: u64,
    /// Minimum scheduler IO batches created by one morsel.
    pub io_batches_per_morsel_min: Option<u64>,
    /// Maximum scheduler IO batches created by one morsel.
    pub io_batches_per_morsel_max: u64,
    /// Maximum exact-ticket suspensions returned by one morsel.
    pub io_blocks_per_morsel_max: u64,
    /// Time to the first batch emitted by this thread.
    pub time_to_first_batch: Option<Duration>,
}

impl ScanStats {
    /// Fold another thread's counters into this one.
    pub fn merge(&mut self, other: &ScanStats) {
        self.morsels += other.morsels;
        self.io_uses += other.io_uses;
        self.io_requests += other.io_requests;
        self.io_batches += other.io_batches;
        self.io_cell_hits += other.io_cell_hits;
        self.io_registered += other.io_registered;
        self.io_bytes += other.io_bytes;
        self.io_waits += other.io_waits;
        self.nowait_attempts += other.nowait_attempts;
        self.nowait_hits += other.nowait_hits;
        self.nowait_misses += other.nowait_misses;
        self.nowait_unsupported += other.nowait_unsupported;
        self.io_wait_time += other.io_wait_time;
        self.decodes += other.decodes;
        self.decode_reuses += other.decode_reuses;
        self.conjuncts_short_circuited += other.conjuncts_short_circuited;
        self.morsels_empty += other.morsels_empty;
        self.execute_io_blocks += other.execute_io_blocks;
        self.morsels_blocked_for_io += other.morsels_blocked_for_io;
        self.io_uses_per_morsel_min =
            min_option(self.io_uses_per_morsel_min, other.io_uses_per_morsel_min);
        self.io_uses_per_morsel_max = self
            .io_uses_per_morsel_max
            .max(other.io_uses_per_morsel_max);
        self.io_requests_per_morsel_min = min_option(
            self.io_requests_per_morsel_min,
            other.io_requests_per_morsel_min,
        );
        self.io_requests_per_morsel_max = self
            .io_requests_per_morsel_max
            .max(other.io_requests_per_morsel_max);
        self.io_batches_per_morsel_min = min_option(
            self.io_batches_per_morsel_min,
            other.io_batches_per_morsel_min,
        );
        self.io_batches_per_morsel_max = self
            .io_batches_per_morsel_max
            .max(other.io_batches_per_morsel_max);
        self.io_blocks_per_morsel_max = self
            .io_blocks_per_morsel_max
            .max(other.io_blocks_per_morsel_max);
        self.time_to_first_batch = match (self.time_to_first_batch, other.time_to_first_batch) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }

    /// Record the scheduling shape of one completed morsel.
    pub(crate) fn record_morsel_io(&mut self, uses: u64, requests: u64, batches: u64, blocks: u64) {
        self.io_uses_per_morsel_min = min_option(self.io_uses_per_morsel_min, Some(uses));
        self.io_uses_per_morsel_max = self.io_uses_per_morsel_max.max(uses);
        self.io_requests_per_morsel_min =
            min_option(self.io_requests_per_morsel_min, Some(requests));
        self.io_requests_per_morsel_max = self.io_requests_per_morsel_max.max(requests);
        self.io_batches_per_morsel_min = min_option(self.io_batches_per_morsel_min, Some(batches));
        self.io_batches_per_morsel_max = self.io_batches_per_morsel_max.max(batches);
        self.io_blocks_per_morsel_max = self.io_blocks_per_morsel_max.max(blocks);
        if blocks > 0 {
            self.morsels_blocked_for_io += 1;
        }
    }
}

fn min_option(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    }
}
