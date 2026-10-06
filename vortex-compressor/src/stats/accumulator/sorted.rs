// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sortedness.

use num_traits::PrimInt;

use super::CHUNK;
use super::IntAccumulator;
use super::Nulls;

/// Whether the valid values are sorted, non-strictly and strictly, in ascending order.
#[derive(Debug, Clone, Copy)]
pub struct Sorted<T> {
    /// The last value seen, if any.
    prev: Option<T>,
    /// The number of adjacent pairs that decrease.
    decreases: u32,
    /// The number of adjacent pairs that are equal.
    repeats: u32,
}

impl<T: PrimInt> Sorted<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: None,
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

    // A filled null equals its neighbour, adding exactly one equal pair, which `unfill` removes.
    const NULLS: Nulls = Nulls::Fill;

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        // Count rather than short-circuit, so both reductions stay branch-free and vectorize.
        // The pair spanning the previous chunk is counted separately, so the rest stays a plain
        // zip that vectorizes.
        let pairs = || values.iter().zip(&values[1..]);
        let prev = self.prev.unwrap_or(values[0]);
        let first_repeat = u8::from(values[0] == prev && self.prev.is_some());
        let decreases =
            u8::from(values[0] < prev) + pairs().map(|(a, b)| u8::from(b < a)).sum::<u8>();
        let repeats = first_repeat + pairs().map(|(a, b)| u8::from(b == a)).sum::<u8>();
        self.decreases += u32::from(decreases);
        self.repeats += u32::from(repeats);
        self.prev = Some(values[CHUNK - 1]);
    }

    #[inline(always)]
    fn unfill(&mut self, _filled: &[T; CHUNK], valid: u64) {
        self.repeats -= (!valid).count_ones();
    }

    #[inline]
    fn finish(self) -> SortedResult {
        SortedResult {
            sorted: self.decreases == 0,
            strict_sorted: self.decreases == 0 && self.repeats == 0,
        }
    }
}

int_stat!(Sorted, SortedStat: SortedResult, |sorted| sorted);
