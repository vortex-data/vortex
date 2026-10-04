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
use super::LANES;
use super::fold_lanes2;
use super::reduce_lanes;

/// The minimum and maximum valid values.
#[derive(Debug, Clone, Copy)]
pub struct MinMax<T> {
    /// The minimum so far of each lane.
    min: [T; LANES],
    /// The maximum so far of each lane.
    max: [T; LANES],
}

impl<T: PrimInt> MinMax<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            min: [T::max_value(); LANES],
            max: [T::min_value(); LANES],
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

    const USES_FILL: bool = true;

    #[inline(always)]
    fn start(&mut self, _head: T) {}

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        fold_lanes2(
            &mut self.min,
            &mut self.max,
            values,
            |min, v| if v < min { v } else { min },
            |max, v| if v > max { v } else { max },
        );
    }

    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], _valid: u64) {
        self.chunk(filled);
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.min[0] = self.min[0].min(value);
        self.max[0] = self.max[0].max(value);
    }

    #[inline]
    fn finish(self) -> (T, T) {
        (
            reduce_lanes(&self.min, T::max_value(), T::min),
            reduce_lanes(&self.max, T::min_value(), T::max),
        )
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
