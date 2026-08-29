// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-run counters. The eval matrix in the prototype plan records these per row.

use std::sync::OnceLock;
use std::time::Duration;

static PUSH_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn push_profile_enabled() -> bool {
    cfg!(feature = "_test-harness")
        && *PUSH_PROFILE_ENABLED.get_or_init(|| {
            std::env::var_os("VORTEX_MORSEL_PUSH_PROFILE").is_some_and(|value| value != "0")
        })
}

/// Counters accumulated by one driving thread, summed across threads at the end of a run.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    /// Push-node events consumed by worker-local pipelines.
    pub push_node_transitions: u64,
    /// Child batches routed directly through a parent edge in the current push trampoline.
    pub push_inline_transfers: u64,
    /// Push continuations queued because an inline trampoline yielded or exhausted its quantum.
    pub push_dispatch_spills: u64,
    /// Ready physical pipelines selected by an affinity-owned morsel driver.
    pub push_pipeline_runs: u64,
    /// Direct operator-stage invocations made by physical pipeline drivers.
    pub push_pipeline_stage_calls: u64,
    /// Payloads handed directly to the next stage without creating a pipeline frame.
    pub push_fast_stage_transfers: u64,
    /// Invocations that required the cold frame/control path.
    pub push_cold_frame_spills: u64,
    /// Atomic demand-and-activation gates drained inside the physical runtime.
    pub push_inline_gates: u64,
    /// Selection-mask clones performed by runtime routing.
    pub push_runtime_mask_clones: u64,
    /// Benchmark-only cumulative time inside the physical runtime poll loop.
    pub push_profile_runtime_time: Duration,
    /// Benchmark-only cumulative time assembling and activating deferred sources.
    pub push_profile_activation_time: Duration,
    /// Benchmark-only cumulative time converting root batches for output.
    pub push_profile_root_time: Duration,
    /// Benchmark-only Flat stage calls and cumulative time.
    pub push_profile_flat: (u64, Duration),
    /// Benchmark-only Chunked stage calls and cumulative time.
    pub push_profile_chunked: (u64, Duration),
    /// Benchmark-only Struct stage calls and cumulative time.
    pub push_profile_struct: (u64, Duration),
    /// Benchmark-only Conjunct stage calls and cumulative time.
    pub push_profile_conjunct: (u64, Duration),
    /// Benchmark-only Filter stage calls and cumulative time.
    pub push_profile_filter: (u64, Duration),
    /// Benchmark-only uncategorized stage calls and cumulative time.
    pub push_profile_other: (u64, Duration),
    /// Benchmark-only Flat decode calls and cumulative time.
    pub push_profile_flat_decode: (u64, Duration),
    /// Benchmark-only Flat selection-filter calls and cumulative time.
    pub push_profile_flat_filter: (u64, Duration),
    /// Parked pipelines made ready by I/O, credit, or cross-pipeline boundary control.
    pub push_pipeline_boundary_resumes: u64,
    /// Flat source activations seeded for push morsels.
    pub push_source_activations: u64,
    /// Root batches emitted by push execution.
    pub push_root_batches: u64,
    /// Stale or duplicate generation-tagged push wakes ignored.
    pub push_stale_wakes: u64,
    /// Largest worker-local ready event deque observed.
    pub push_ready_events_max: u64,
    /// Sliding filtered-lookahead window refills after morsel completion.
    pub lookahead_refills: u64,
    /// Optional demand hints published by push operators.
    pub demand_hints_emitted: u64,
    /// Demand hints observed by the scheduler after delivery policy.
    pub demand_hints_observed: u64,
    /// Demand hints intentionally disabled or left delayed at morsel retirement.
    pub demand_hints_dropped: u64,
    /// Deferred, unissued source reads examined after target/range catalog pruning.
    pub demand_io_candidates: u64,
    /// Deferred exact reads admitted after a nonempty demand hint.
    pub demand_io_promotions: u64,
    /// Matching deferred reads left unissued after an all-false demand hint.
    pub demand_io_suppressed: u64,
    /// Largest number of disjoint monotone demand spans retained by one worker-local morsel.
    pub demand_state_live_max: u64,
    /// Maximum selected rows holding root output credit.
    pub output_rows_max: u64,
    /// Maximum array bytes holding root output credit.
    pub output_bytes_max: u64,
    /// Root batches that waited for output credit or the ordering frontier.
    pub output_credit_blocks: u64,
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
        self.push_node_transitions += other.push_node_transitions;
        self.push_inline_transfers += other.push_inline_transfers;
        self.push_dispatch_spills += other.push_dispatch_spills;
        self.push_pipeline_runs += other.push_pipeline_runs;
        self.push_pipeline_stage_calls += other.push_pipeline_stage_calls;
        self.push_fast_stage_transfers += other.push_fast_stage_transfers;
        self.push_cold_frame_spills += other.push_cold_frame_spills;
        self.push_inline_gates += other.push_inline_gates;
        self.push_runtime_mask_clones += other.push_runtime_mask_clones;
        self.push_profile_runtime_time += other.push_profile_runtime_time;
        self.push_profile_activation_time += other.push_profile_activation_time;
        self.push_profile_root_time += other.push_profile_root_time;
        merge_profile(&mut self.push_profile_flat, other.push_profile_flat);
        merge_profile(&mut self.push_profile_chunked, other.push_profile_chunked);
        merge_profile(&mut self.push_profile_struct, other.push_profile_struct);
        merge_profile(&mut self.push_profile_conjunct, other.push_profile_conjunct);
        merge_profile(&mut self.push_profile_filter, other.push_profile_filter);
        merge_profile(&mut self.push_profile_other, other.push_profile_other);
        merge_profile(
            &mut self.push_profile_flat_decode,
            other.push_profile_flat_decode,
        );
        merge_profile(
            &mut self.push_profile_flat_filter,
            other.push_profile_flat_filter,
        );
        self.push_pipeline_boundary_resumes += other.push_pipeline_boundary_resumes;
        self.push_source_activations += other.push_source_activations;
        self.push_root_batches += other.push_root_batches;
        self.push_stale_wakes += other.push_stale_wakes;
        self.push_ready_events_max = self.push_ready_events_max.max(other.push_ready_events_max);
        self.lookahead_refills += other.lookahead_refills;
        self.demand_hints_emitted += other.demand_hints_emitted;
        self.demand_hints_observed += other.demand_hints_observed;
        self.demand_hints_dropped += other.demand_hints_dropped;
        self.demand_io_candidates += other.demand_io_candidates;
        self.demand_io_promotions += other.demand_io_promotions;
        self.demand_io_suppressed += other.demand_io_suppressed;
        self.demand_state_live_max = self.demand_state_live_max.max(other.demand_state_live_max);
        self.output_rows_max = self.output_rows_max.max(other.output_rows_max);
        self.output_bytes_max = self.output_bytes_max.max(other.output_bytes_max);
        self.output_credit_blocks += other.output_credit_blocks;
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

fn merge_profile(into: &mut (u64, Duration), other: (u64, Duration)) {
    into.0 += other.0;
    into.1 += other.1;
}

fn min_option(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    }
}
