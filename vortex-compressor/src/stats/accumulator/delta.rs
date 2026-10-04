// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The range of differences between consecutive values.

use num_traits::Bounded;

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::IntValue;
use super::LANES;
use super::fold_lanes2;

/// The smallest and largest exact difference between consecutive valid values.
///
/// Equal bounds mean the valid values form an arithmetic sequence. Nulls are skipped, so the
/// differences are between valid values that are consecutive among the valid values.
#[derive(Debug, Clone, Copy)]
pub struct DeltaRange<T: IntValue> {
    /// The last value seen.
    prev: T,
    /// Whether `prev` holds a value, so the next value has a difference.
    started: bool,
    /// Whether any difference was seen.
    has_delta: bool,
    /// The smallest difference so far of each lane.
    min: [T::Delta; LANES],
    /// The largest difference so far of each lane.
    max: [T::Delta; LANES],
}

impl<T: IntValue> DeltaRange<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: T::zero(),
            started: false,
            has_delta: false,
            min: [Bounded::max_value(); LANES],
            max: [Bounded::min_value(); LANES],
        }
    }
}

impl<T: IntValue> Default for DeltaRange<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: IntValue> IntAccumulator<T> for DeltaRange<T> {
    /// `(min, max)` of the differences, or `None` with fewer than two valid values.
    type Output = Option<(i128, i128)>;

    // The head is fed again after `start`, so it is handled by `started` instead.
    #[inline(always)]
    fn start(&mut self, _head: T) {}

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        if size_of::<T::Delta>() > 8 {
            // Differences of 64-bit values need an `i128`, which does not vectorize.
            for &value in values {
                self.push(value);
            }
            return;
        }
        let mut deltas: [T::Delta; CHUNK] = std::array::from_fn(|i| {
            if i == 0 {
                values[0].delta(self.prev)
            } else {
                values[i].delta(values[i - 1])
            }
        });
        if !self.started {
            // The first value has no predecessor; repeating another difference in its place
            // leaves the extrema unchanged.
            deltas[0] = deltas[1];
        }
        fold_lanes2(
            &mut self.min,
            &mut self.max,
            &deltas,
            |min, d| if d < min { d } else { min },
            |max, d| if d > max { d } else { max },
        );
        self.prev = values[CHUNK - 1];
        self.started = true;
        self.has_delta = true;
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        if self.started {
            let delta = value.delta(self.prev);
            self.min[0] = self.min[0].min(delta);
            self.max[0] = self.max[0].max(delta);
            self.has_delta = true;
        }
        self.prev = value;
        self.started = true;
    }

    #[inline]
    fn finish(self) -> Option<(i128, i128)> {
        self.has_delta.then(|| {
            (
                self.min
                    .into_iter()
                    .map(Into::into)
                    .min()
                    .unwrap_or(i128::MAX),
                self.max
                    .into_iter()
                    .map(Into::into)
                    .max()
                    .unwrap_or(i128::MIN),
            )
        })
    }
}

/// The key of the [`DeltaRange`] statistic.
pub struct DeltaRangeStat;

impl IntStat for DeltaRangeStat {
    type Value = Option<(i128, i128)>;
}

impl<T: IntValue> ErasedAccumulator<T> for DeltaRange<T> {
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<DeltaRangeStat>(self.finish());
    }
}
