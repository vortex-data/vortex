// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic simulation test harness for `vortex-io` and downstream crates.
//!
//! Exposed under `#[cfg(any(test, feature = "_test-harness"))]`; not part of the public API.

mod executor;
mod read_at;

pub use executor::SimulatedExecutor;
pub use read_at::PendingReadId;
pub use read_at::ReadFault;
pub use read_at::SimulatedReadAt;
use vortex_error::VortexExpect;

/// Lightweight, dependency-free SplitMix64 PRNG for deterministic test scheduling.
#[derive(Clone, Debug)]
pub(crate) struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn next_usize(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let bound_u64 = u64::try_from(bound).vortex_expect("usize fits in u64");
        usize::try_from(self.next_u64() % bound_u64).vortex_expect("modulo result fits in usize")
    }

    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}
