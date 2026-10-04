// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The exact sum.

use std::marker::PhantomData;

use num_traits::AsPrimitive;
use num_traits::PrimInt;

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::IntValue;
use super::LANES;
use super::fill_nulls;
use super::fold_lanes;
use super::is_mostly_valid;
use super::push_set_bits;

/// The exact sum of the valid values. Any `u32::MAX` 64-bit values sum within an `i128`.
#[derive(Debug, Clone, Copy)]
pub struct Sum<T> {
    /// The sum so far.
    total: i128,
    /// Per-lane sums of values of at most 32 bits. Each lane sums at most `u32::MAX / LANES`
    /// values, so cannot overflow.
    lanes: [i64; LANES],
    /// The summed type.
    _type: PhantomData<T>,
}

impl<T> Sum<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            total: 0,
            lanes: [0; LANES],
            _type: PhantomData,
        }
    }
}

impl<T> Default for Sum<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sums a chunk of 64-bit values exactly, in lanes narrow enough to vectorize.
#[inline(always)]
fn wide_chunk_sum<T>(values: &[T; CHUNK]) -> i128
where
    T: PrimInt + AsPrimitive<i64> + AsPrimitive<u64>,
{
    if T::min_value() < T::zero() {
        // Split each value into a signed high half and an unsigned low half, whose sums over a
        // chunk fit in an `i64`.
        let (mut high, mut low) = (0i64, 0i64);
        for &v in values {
            let v: i64 = v.as_();
            high += v >> 32;
            low += v & 0xFFFF_FFFF;
        }
        (i128::from(high) << 32) + i128::from(low)
    } else {
        let (mut high, mut low) = (0u64, 0u64);
        for &v in values {
            let v: u64 = v.as_();
            high += v >> 32;
            low += v & 0xFFFF_FFFF;
        }
        (i128::from(high) << 32) + i128::from(low)
    }
}

impl<T> IntAccumulator<T> for Sum<T>
where
    T: IntValue,
{
    type Output = i128;

    #[inline(always)]
    fn start(&mut self, _head: T) {}

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        if size_of::<T>() <= 4 {
            fold_lanes(&mut self.lanes, values, |acc, v| {
                acc + AsPrimitive::<i64>::as_(v)
            });
        } else {
            self.total += wide_chunk_sum(values);
        }
    }

    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        if is_mostly_valid(valid) {
            self.chunk(&fill_nulls(values, valid, T::zero()));
        } else {
            push_set_bits(self, values, valid);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.total += AsPrimitive::<i128>::as_(value);
    }

    #[inline]
    fn finish(self) -> i128 {
        self.total
            + self
                .lanes
                .iter()
                .map(|&lane| i128::from(lane))
                .sum::<i128>()
    }
}

/// The key of the [`Sum`] statistic.
pub struct SumStat;

impl IntStat for SumStat {
    type Value = i128;
}

impl<T> ErasedAccumulator<T> for Sum<T>
where
    T: IntValue,
{
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<SumStat>(self.finish());
    }
}
