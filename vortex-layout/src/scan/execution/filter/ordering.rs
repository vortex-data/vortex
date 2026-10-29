// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recent predicate costs and occasional alternative orders for Scan V2 chunks.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bit_vec::BitVec;
use parking_lot::Mutex;
use smallvec::SmallVec;
use vortex_error::VortexExpect;

const MIN_OBSERVATIONS: usize = 4;
const RECENT_WEIGHT: f64 = 0.25;
const SWITCH_RATIO: f64 = 0.9;
const EXPLORATION_INTERVAL: u64 = 32;

type Indices = SmallVec<[usize; 4]>;

pub(super) struct AdaptiveOrder(Mutex<OrderState>);

pub(super) struct ChunkOrder {
    pub(super) indices: Indices,
    pub(super) exploration: bool,
}

impl ChunkOrder {
    pub(super) fn next_conjunct(&self, remaining: &BitVec) -> Option<usize> {
        self.indices.iter().copied().find(|&index| remaining[index])
    }
}

pub(super) struct Observation {
    pub(super) version: u64,
    pub(super) input_rows: usize,
    pub(super) output_rows: usize,
    pub(super) compute_ns: u128,
    pub(super) recheck: bool,
}

impl AdaptiveOrder {
    pub(super) fn new(conjuncts: usize) -> Self {
        static NEXT_SEED: AtomicU64 = AtomicU64::new(0);
        Self(Mutex::new(OrderState::new(
            conjuncts,
            NEXT_SEED.fetch_add(1, Ordering::Relaxed),
        )))
    }

    pub(super) fn begin_chunk(&self) -> ChunkOrder {
        self.0.lock().begin_chunk()
    }

    pub(super) fn version_changed(&self, index: usize, version: u64) {
        self.0.lock().version_changed(index, version);
    }

    pub(super) fn report(&self, index: usize, observation: Observation) {
        // Rechecks see rows that already passed this predicate. Their conditional selectivity
        // cannot predict the cost or rejection rate of a first evaluation on a new chunk.
        if observation.recheck || observation.input_rows == 0 {
            return;
        }
        let mut state = self.0.lock();
        if state.estimates.len() < 2 || observation.version < state.estimates[index].version {
            return;
        }
        state.version_changed(index, observation.version);
        state.estimates[index].observe(observation);
        state.reorder();
    }
}

struct OrderState {
    preferred: Indices,
    estimates: Vec<Estimate>,
    random: u64,
}

impl OrderState {
    fn new(conjuncts: usize, seed: u64) -> Self {
        Self {
            preferred: (0..conjuncts).collect(),
            estimates: (0..conjuncts).map(|_| Estimate::default()).collect(),
            random: seed,
        }
    }

    fn begin_chunk(&mut self) -> ChunkOrder {
        let mut indices = self.preferred.clone();
        let exploration =
            indices.len() > 1 && self.next_random().is_multiple_of(EXPLORATION_INTERVAL);
        if exploration {
            let left = usize::try_from(self.next_random() % (indices.len() - 1) as u64)
                .vortex_expect("exploration index is smaller than the conjunct count");
            indices.swap(left, left + 1);
        }
        ChunkOrder {
            indices,
            exploration,
        }
    }

    fn next_random(&mut self) -> u64 {
        // SplitMix64 disperses adjacent per-file seeds and avoids synchronizing exploration
        // across files or workers. This generator is only used under the ordering lock.
        self.random = self.random.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.random;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn version_changed(&mut self, index: usize, version: u64) {
        let estimate = &mut self.estimates[index];
        if version > estimate.version {
            tracing::debug!(
                conjunct = index,
                previous_version = estimate.version,
                version,
                "v2 predicate estimates invalidated"
            );
            estimate.version = version;
            estimate.observations = 0;
        }
    }

    fn reorder(&mut self) {
        if self
            .estimates
            .iter()
            .any(|estimate| estimate.observations < MIN_OBSERVATIONS)
        {
            return;
        }
        let mut candidate = self.preferred.clone();
        candidate.sort_by(|&left, &right| {
            self.estimates[left]
                .cost_per_rejected_row()
                .total_cmp(&self.estimates[right].cost_per_rejected_row())
        });
        if candidate == self.preferred
            || self.expected_cost(&candidate) >= self.expected_cost(&self.preferred) * SWITCH_RATIO
        {
            return;
        }
        tracing::debug!(
            previous = ?self.preferred,
            order = ?candidate,
            estimated_before_ns = self.expected_cost(&self.preferred),
            estimated_after_ns = self.expected_cost(&candidate),
            "v2 predicate order changed"
        );
        self.preferred = candidate;
    }

    fn expected_cost(&self, order: &[usize]) -> f64 {
        let mut surviving = 1.0;
        let mut cost = 0.0;
        for &index in order {
            let estimate = &self.estimates[index];
            cost += surviving * estimate.cost_per_row_ns;
            surviving *= estimate.pass_fraction;
        }
        cost
    }
}

#[derive(Default)]
struct Estimate {
    version: u64,
    observations: usize,
    cost_per_row_ns: f64,
    pass_fraction: f64,
}

impl Estimate {
    fn observe(&mut self, observation: Observation) {
        let cost = observation.compute_ns.max(1) as f64 / observation.input_rows as f64;
        let passed = observation.output_rows as f64 / observation.input_rows as f64;
        if self.observations == 0 {
            self.cost_per_row_ns = cost;
            self.pass_fraction = passed;
        } else {
            self.cost_per_row_ns += RECENT_WEIGHT * (cost - self.cost_per_row_ns);
            self.pass_fraction += RECENT_WEIGHT * (passed - self.pass_fraction);
        }
        self.observations = self.observations.saturating_add(1);
    }

    fn cost_per_rejected_row(&self) -> f64 {
        self.cost_per_row_ns / (1.0 - self.pass_fraction).max(f64::EPSILON)
    }
}

#[cfg(test)]
mod tests;
