// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::CHUNK;
use super::ChunkAccumulator;
use super::low_bits;

/// Whether the values are sorted in increasing order, strictly if `STRICT`, with nulls first.
///
/// An array of only nulls is sorted but not strictly sorted.
pub struct IsSorted<T, const STRICT: bool> {
    /// The last valid value.
    prev: Option<T>,
    /// Whether the values so far are sorted.
    sorted: bool,
    /// The number of nulls before the first valid value.
    leading_nulls: u32,
}

impl<T: Copy + Ord, const STRICT: bool> Default for IsSorted<T, STRICT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord, const STRICT: bool> IsSorted<T, STRICT> {
    /// Returns an accumulator that has seen no values.
    pub fn new() -> Self {
        Self {
            prev: None,
            sorted: true,
            leading_nulls: 0,
        }
    }

    /// Returns whether the values are sorted.
    pub fn finish(&self) -> bool {
        // Only nulls are sorted, but not strictly.
        let only_nulls = self.prev.is_none() && self.leading_nulls > 0;
        self.sorted && !(STRICT && only_nulls)
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn in_order(a: T, b: T) -> bool {
        if STRICT { a < b } else { a <= b }
    }

    /// Checks a run of valid values, which follows the previous valid value.
    ///
    /// Without 64-bit vector compares in the baseline instruction set, 64-bit values compare one
    /// at a time and stop at the first disorder. Otherwise each chunk compares branch-free, so
    /// that the comparison vectorizes, and the first disordered chunk stops the comparison.
    /// `size_of` and `cfg!` are constants, so the choice folds.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn run(&mut self, values: &[T]) {
        let (prev, rest) = match self.prev {
            Some(prev) => (prev, values),
            None => (values[0], &values[1..]),
        };
        let ordered = if size_of::<T>() == 8 && !cfg!(target_feature = "avx2") {
            Self::ordered_scalar(prev, rest)
        } else {
            Self::ordered_chunks(prev, rest)
        };
        self.sorted &= ordered;
        self.prev = values.last().copied();
    }

    /// Whether `prev` and then `values` are in order, comparing one value at a time.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn ordered_scalar(mut prev: T, values: &[T]) -> bool {
        for &value in values {
            if !Self::in_order(prev, value) {
                return false;
            }
            prev = value;
        }
        true
    }

    /// Whether `prev` and then `values` are in order, comparing a chunk at a time.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn ordered_chunks(prev: T, values: &[T]) -> bool {
        let Some((&first, _)) = values.split_first() else {
            return true;
        };
        if !Self::in_order(prev, first) {
            return false;
        }
        values
            .chunks(CHUNK)
            .zip(values[1..].chunks(CHUNK))
            .all(|(a, b)| {
                a.iter()
                    .zip(b)
                    .fold(true, |acc, (&a, &b)| acc & Self::in_order(a, b))
            })
    }
}

impl<T: Copy + Ord, const STRICT: bool> ChunkAccumulator<T> for IsSorted<T, STRICT> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.run(values);
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        self.run(chunks.as_flattened());
    }

    fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, len: usize) {
        let nulls = !valid & low_bits(len);
        if nulls != 0 {
            // Nulls sort first, so they may only precede every valid value.
            let first_valid = valid.trailing_zeros();
            if self.prev.is_some() || (first_valid < 64 && nulls >> first_valid != 0) {
                self.sorted = false;
                return;
            }
            self.leading_nulls += nulls.count_ones();
            if STRICT && self.leading_nulls > 1 {
                self.sorted = false;
                return;
            }
        }
        if valid != 0 {
            self.run(&values[valid.trailing_zeros() as usize..len]);
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn is_done(&self) -> bool {
        !self.sorted
    }
}
