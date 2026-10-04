// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The number of runs.

use num_traits::PrimInt;

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::transitions;

/// The number of runs of equal consecutive valid values. Nulls do not break runs.
#[derive(Debug, Clone, Copy)]
pub struct RunCount<T> {
    /// The last value seen.
    prev: T,
    /// The number of runs so far.
    runs: u32,
}

impl<T: PrimInt> RunCount<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: T::zero(),
            runs: 0,
        }
    }
}

impl<T: PrimInt> Default for RunCount<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PrimInt> IntAccumulator<T> for RunCount<T> {
    type Output = u32;

    const USES_FILL: bool = true;

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.prev = head;
        self.runs = 1;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.runs += transitions(&self.prev, values);
        self.prev = values[CHUNK - 1];
    }

    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], _valid: u64) {
        // Filled nulls repeat their predecessor, so they add no runs.
        self.chunk(filled);
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.runs += u32::from(value != self.prev);
        self.prev = value;
    }

    #[inline]
    fn finish(self) -> u32 {
        self.runs
    }
}

/// The key of the [`RunCount`] statistic.
pub struct RunCountStat;

impl IntStat for RunCountStat {
    type Value = u32;
}

impl<T: PrimInt> ErasedAccumulator<T> for RunCount<T> {
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<RunCountStat>(self.finish());
    }
}
