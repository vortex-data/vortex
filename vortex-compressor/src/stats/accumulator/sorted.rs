// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sortedness.

use num_traits::PrimInt;

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::forward_fill;
use super::is_mostly_valid;
use super::push_set_bits;

/// Whether the valid values are sorted, non-strictly and strictly, in ascending order.
#[derive(Debug, Clone, Copy)]
pub struct Sorted<T> {
    /// The last value seen.
    prev: T,
    /// The number of adjacent pairs that decrease.
    decreases: u32,
    /// The number of adjacent pairs that are equal.
    repeats: u32,
}

impl<T: PrimInt> Sorted<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: T::zero(),
            decreases: 0,
            repeats: 0,
        }
    }
}

impl<T: PrimInt> Default for Sorted<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether values are sorted, as computed by [`Sorted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortedResult {
    /// Every value is at least the previous one.
    pub sorted: bool,
    /// Every value is greater than the previous one.
    pub strict_sorted: bool,
}

impl<T: PrimInt> IntAccumulator<T> for Sorted<T> {
    type Output = SortedResult;

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.prev = head;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        // Count rather than short-circuit, so both reductions stay branch-free and vectorize.
        // The pair spanning the previous chunk is counted separately, so the rest stays a plain
        // zip that vectorizes.
        let pairs = || values.iter().zip(&values[1..]);
        let decreases =
            u8::from(values[0] < self.prev) + pairs().map(|(a, b)| u8::from(b < a)).sum::<u8>();
        let repeats =
            u8::from(values[0] == self.prev) + pairs().map(|(a, b)| u8::from(b == a)).sum::<u8>();
        self.decreases += u32::from(decreases);
        self.repeats += u32::from(repeats);
        self.prev = values[CHUNK - 1];
    }

    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        if is_mostly_valid(valid) {
            self.chunk(&forward_fill(values, valid, self.prev));
            // Each filled null repeats the value before it, adding exactly one equal pair.
            self.repeats -= (!valid).count_ones();
        } else {
            push_set_bits(self, values, valid);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.decreases += u32::from(value < self.prev);
        self.repeats += u32::from(value == self.prev);
        self.prev = value;
    }

    #[inline]
    fn finish(self) -> SortedResult {
        // `start` and the first `push` or `chunk` compare the head with itself once.
        let repeats = self.repeats.saturating_sub(1);
        SortedResult {
            sorted: self.decreases == 0,
            strict_sorted: self.decreases == 0 && repeats == 0,
        }
    }
}

/// The key of the [`Sorted`] statistic.
pub struct SortedStat;

impl IntStat for SortedStat {
    type Value = SortedResult;
}

impl<T: PrimInt> ErasedAccumulator<T> for Sorted<T> {
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<SortedStat>(self.finish());
    }
}
