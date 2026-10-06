// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Bounded;

use super::CHUNK;
use super::ChunkAccumulator;
use super::forward_fill;

/// The smallest and largest valid integer.
///
/// Each lane keeps its own bounds, which are reduced once at the end.
pub struct MinMax<T> {
    /// The minimum so far of each lane.
    min: [T; CHUNK],
    /// The maximum so far of each lane.
    max: [T; CHUNK],
    /// Whether any value was valid.
    any_valid: bool,
}

impl<T: Copy + Ord + Bounded> Default for MinMax<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord + Bounded> MinMax<T> {
    /// Returns an accumulator that has seen no values.
    pub fn new() -> Self {
        Self {
            min: [T::max_value(); CHUNK],
            max: [T::min_value(); CHUNK],
            any_valid: false,
        }
    }

    /// Returns the smallest and largest valid value, or `None` if no value was valid.
    pub fn finish(&self) -> Option<(T, T)> {
        self.any_valid.then(|| {
            let min = self.min.iter().fold(T::max_value(), |acc, &v| acc.min(v));
            let max = self.max.iter().fold(T::min_value(), |acc, &v| acc.max(v));
            (min, max)
        })
    }

    /// Folds a chunk into the bounds of each lane.
    ///
    /// The lane count depends on the width and the instruction set, and was chosen by
    /// measurement. With too few lanes for the vector width, the vectorizer packs lanes across
    /// groups and loads them one by one. Without 64-bit vector compares in the baseline instruction
    /// set, 64-bit values use only a few lanes. `size_of` and `cfg!` are constants, so the dispatch
    /// folds.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn fold(&mut self, values: &[T; CHUNK]) {
        let avx2 = cfg!(target_feature = "avx2");
        match size_of::<T>() {
            1 => self.lanes::<32>(values),
            2 => self.lanes::<CHUNK>(values),
            4 if avx2 => self.lanes::<CHUNK>(values),
            4 => self.lanes::<16>(values),
            _ if avx2 => self.lanes::<8>(values),
            _ => self.lanes::<4>(values),
        }
    }

    /// Folds a chunk into the bounds of the first `L` lanes.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn lanes<const L: usize>(&mut self, values: &[T; CHUNK]) {
        let (Some(min), Some(max)) = (
            self.min.first_chunk_mut::<L>(),
            self.max.first_chunk_mut::<L>(),
        ) else {
            unreachable!("L is at most CHUNK")
        };
        for group in values.as_chunks::<L>().0 {
            for i in 0..L {
                min[i] = if group[i] < min[i] { group[i] } else { min[i] };
                max[i] = if group[i] > max[i] { group[i] } else { max[i] };
            }
        }
    }
}

impl<T: Copy + Ord + Bounded> ChunkAccumulator<T> for MinMax<T> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.fold(values);
        self.any_valid = true;
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, _len: usize) {
        if valid == 0 {
            return;
        }
        // Nulls take a valid value of the chunk, which changes neither bound.
        let fill = values[valid.trailing_zeros() as usize];
        self.fold(&forward_fill(values, valid, fill));
        self.any_valid = true;
    }
}
