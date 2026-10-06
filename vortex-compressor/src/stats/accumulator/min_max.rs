// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The minimum and maximum.

use num_traits::PrimInt;
use vortex_array::scalar::PValue;

use super::CHUNK;
use super::IntAccumulator;
use super::LANES;
use super::Nulls;
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

    // Repeating a valid value changes neither extreme.
    const NULLS: Nulls = Nulls::Fill;

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

    #[inline]
    fn finish(self) -> (T, T) {
        (
            reduce_lanes(&self.min, T::max_value(), T::min),
            reduce_lanes(&self.max, T::min_value(), T::max),
        )
    }
}

int_stat!(MinMax, MinMaxStat: (PValue, PValue), |(min, max): (T, T)| (
    min.to_pvalue(),
    max.to_pvalue()
));
