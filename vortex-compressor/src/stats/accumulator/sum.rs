// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The exact sum.

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use super::CHUNK;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::IntValue;
use super::LANES;
use super::fill_nulls;
use super::fold_lanes;
use super::fold_lanes2;
use super::is_mostly_valid;
use super::push_set_bits;

/// The exact sum of the valid values.
///
/// Values are summed in the narrowest lanes that cannot overflow within a block, which are added
/// into an `i128` total after each block.
#[derive(Debug, Clone, Copy)]
pub struct Sum<T> {
    /// The sum of the flushed lanes and pushed values.
    total: i128,
    /// Lanes for values of at most 16 bits. A block has 8 lanes of at most 512 16-bit values, so a
    /// lane stays below `2^25`.
    narrow: [i32; LANES],
    /// Lanes for 32-bit values, or the low halves of 64-bit values.
    low: [i64; LANES],
    /// Lanes for the high halves of 64-bit values.
    high: [i64; LANES],
    /// The summed type.
    _type: PhantomData<T>,
}

impl<T> Sum<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            total: 0,
            narrow: [0; LANES],
            low: [0; LANES],
            high: [0; LANES],
            _type: PhantomData,
        }
    }
}

impl<T> Default for Sum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: IntValue> IntAccumulator<T> for Sum<T> {
    type Output = i128;

    #[inline(always)]
    fn start(&mut self, _head: T) {}

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        match size_of::<T>() {
            ..=2 => fold_lanes(&mut self.narrow, values, |acc, v| {
                acc + AsPrimitive::<i32>::as_(v)
            }),
            4 => fold_lanes(&mut self.low, values, |acc, v| {
                acc + AsPrimitive::<i64>::as_(v)
            }),
            // Split each value into a high half, signed for signed types, and an unsigned low
            // half, whose sums over a block fit in an `i64`.
            _ if T::min_value() < T::zero() => fold_lanes2(
                &mut self.high,
                &mut self.low,
                values,
                |acc, v| acc + (AsPrimitive::<i64>::as_(v) >> 32),
                |acc, v| acc + (AsPrimitive::<i64>::as_(v) & 0xFFFF_FFFF),
            ),
            _ => fold_lanes2(
                &mut self.high,
                &mut self.low,
                values,
                |acc, v| acc + (AsPrimitive::<u64>::as_(v) >> 32) as i64,
                |acc, v| acc + (AsPrimitive::<u64>::as_(v) & 0xFFFF_FFFF) as i64,
            ),
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

    #[inline(always)]
    fn end_block(&mut self) {
        let narrow: i64 = self.narrow.iter().map(|&lane| i64::from(lane)).sum();
        let low: i128 = self.low.iter().map(|&lane| i128::from(lane)).sum();
        let high: i128 = self.high.iter().map(|&lane| i128::from(lane)).sum();
        self.total += i128::from(narrow) + low + (high << 32);
        self.narrow = [0; LANES];
        self.low = [0; LANES];
        self.high = [0; LANES];
    }

    #[inline]
    fn finish(mut self) -> i128 {
        self.end_block();
        self.total
    }
}

/// The key of the [`Sum`] statistic.
pub struct SumStat;

impl IntStat for SumStat {
    type Value = i128;
}

impl<T: IntValue> ErasedAccumulator<T> for Sum<T> {
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<SumStat>(self.finish());
    }
}
