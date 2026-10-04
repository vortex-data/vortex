// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The minimum and maximum.

use num_traits::PrimInt;
use vortex_array::scalar::PValue;

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::IntValue;
use super::forward_fill;
use super::is_mostly_valid;
use super::push_set_bits;

/// The number of independent lanes that [`MinMax`] reduces into. Keeping the per-lane state
/// across chunks lets the reduction stay in vector registers, with a single horizontal reduction
/// in [`finish`](IntAccumulator::finish).
const MIN_MAX_LANES: usize = 32;

/// The minimum and maximum valid values.
#[derive(Debug, Clone, Copy)]
pub struct MinMax<T> {
    /// The minimum so far of each lane.
    min: [T; MIN_MAX_LANES],
    /// The maximum so far of each lane.
    max: [T; MIN_MAX_LANES],
    /// The first valid value, which stands in for leading nulls in [`forward_fill`].
    head: T,
}

impl<T: PrimInt> MinMax<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            min: [T::max_value(); MIN_MAX_LANES],
            max: [T::min_value(); MIN_MAX_LANES],
            head: T::zero(),
        }
    }

    /// Folds `values` into the lanes.
    #[inline(always)]
    fn fold(&mut self, values: &[T; CHUNK]) {
        // Without 64-bit vector compares in the baseline instruction set, wide lanes spill out of
        // registers, so 64-bit values use fewer lanes. `size_of` is a constant, so this folds.
        if size_of::<T>() >= 8 {
            fold_lanes::<T, 4>(&mut self.min, &mut self.max, values);
        } else {
            fold_lanes::<T, MIN_MAX_LANES>(&mut self.min, &mut self.max, values);
        }
    }
}

impl<T: PrimInt> Default for MinMax<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PrimInt> IntAccumulator<T> for MinMax<T> {
    /// `(min, max)`.
    type Output = (T, T);

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.head = head;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.fold(values);
    }

    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        if is_mostly_valid(valid) {
            self.fold(&forward_fill(values, valid, self.head));
        } else {
            push_set_bits(self, values, valid);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.min[0] = self.min[0].min(value);
        self.max[0] = self.max[0].max(value);
    }

    #[inline]
    fn finish(self) -> (T, T) {
        (
            self.min.into_iter().fold(T::max_value(), T::min),
            self.max.into_iter().fold(T::min_value(), T::max),
        )
    }
}

/// Folds `values` into the first `L` lanes of `min` and `max`.
#[inline(always)]
fn fold_lanes<T: PrimInt, const L: usize>(
    min: &mut [T; MIN_MAX_LANES],
    max: &mut [T; MIN_MAX_LANES],
    values: &[T; CHUNK],
) {
    let (Some(min), Some(max)) = (min.first_chunk_mut::<L>(), max.first_chunk_mut::<L>()) else {
        unreachable!("L is at most MIN_MAX_LANES")
    };
    for group in values.as_chunks::<L>().0 {
        for i in 0..L {
            min[i] = if group[i] < min[i] { group[i] } else { min[i] };
            max[i] = if group[i] > max[i] { group[i] } else { max[i] };
        }
    }
}

/// The key of the [`MinMax`] statistic: `(min, max)`.
pub struct MinMaxStat;

impl IntStat for MinMaxStat {
    type Value = (PValue, PValue);
}

impl<T: IntValue> ErasedAccumulator<T> for MinMax<T> {
    fn finish_into(self, stats: &mut IntStats) {
        let (min, max) = self.finish();
        stats.insert::<MinMaxStat>((min.to_pvalue(), max.to_pvalue()));
    }
}
