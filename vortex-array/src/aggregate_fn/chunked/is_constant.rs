// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::CHUNK;
use super::ChunkAccumulator;
use super::low_bits;

/// Whether every value is the same: either all null, or all valid and equal.
pub struct IsConstant<T> {
    /// The first valid value.
    first: Option<T>,
    /// Whether every valid value so far equals `first`.
    equal: bool,
    /// Whether any value was valid.
    any_valid: bool,
    /// Whether any value was null.
    any_null: bool,
}

impl<T: Copy + Eq> Default for IsConstant<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Eq> IsConstant<T> {
    /// Returns an accumulator that has seen no values.
    pub fn new() -> Self {
        Self {
            first: None,
            equal: true,
            any_valid: false,
            any_null: false,
        }
    }

    /// Returns whether every value is the same, or `None` if there were no values.
    pub fn finish(&self) -> Option<bool> {
        (self.any_valid || self.any_null).then_some(self.equal && !self.mixed())
    }

    /// Whether there were both valid and null values.
    fn mixed(&self) -> bool {
        self.any_valid && self.any_null
    }

    /// Compares valid values with the first.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn compare(&mut self, values: &[T]) {
        let first = *self.first.get_or_insert(values[0]);
        self.any_valid = true;
        // Each chunk compares branch-free, so that the comparison vectorizes, and the first
        // difference stops the comparison.
        for chunk in values.chunks(CHUNK) {
            if !self.equal {
                return;
            }
            self.equal &= chunk.iter().fold(true, |acc, &v| acc & (v == first));
        }
    }
}

impl<T: Copy + Eq> ChunkAccumulator<T> for IsConstant<T> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.compare(values);
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        self.compare(chunks.as_flattened());
    }

    fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, len: usize) {
        if valid == low_bits(len) {
            // The padding of a short chunk repeats its first value, so the whole chunk compares.
            self.compare(values);
            return;
        }
        self.any_null = true;
        self.any_valid |= valid != 0;
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn is_done(&self) -> bool {
        !self.equal || self.mixed()
    }
}
