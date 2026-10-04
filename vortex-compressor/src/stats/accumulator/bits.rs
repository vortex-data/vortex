// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-level statistics. These treat signed values by their two's complement bit pattern, as bit
//! packing does.

use num_traits::AsPrimitive;
use num_traits::PrimInt;

use super::CHUNK;
use super::CHUNK_U32;
use super::ErasedAccumulator;
use super::IntAccumulator;
use super::IntStat;
use super::IntStats;
use super::IntValue;
use super::LANES;
use super::fold_lanes2;
use super::null_indices;

/// Returns the bit pattern of `value`, zero-extended to 64 bits.
#[inline(always)]
fn to_bits<T: PrimInt + AsPrimitive<u64>>(value: T) -> u64 {
    let bits: u64 = value.as_();
    if size_of::<T>() == 8 {
        bits
    } else {
        // `as` sign-extends signed values; keep only the type's own bits.
        bits & ((1u64 << (8 * size_of::<T>())) - 1)
    }
}

/// The bitwise AND and OR of the valid values.
#[derive(Debug, Clone, Copy)]
pub struct CommonBits<T> {
    /// The AND so far of each lane.
    and: [T; LANES],
    /// The OR so far of each lane.
    or: [T; LANES],
}

impl<T: PrimInt> CommonBits<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            and: [!T::zero(); LANES],
            or: [T::zero(); LANES],
        }
    }
}

impl<T: PrimInt> Default for CommonBits<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The bits that the valid values share, as computed by [`CommonBits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonBitsResult {
    /// The bits set in every value, zero-extended to 64 bits.
    pub and: u64,
    /// The bits set in any value, zero-extended to 64 bits.
    pub or: u64,
    /// The width of the type in bits.
    pub width: u32,
}

impl CommonBitsResult {
    /// The bits needed to represent every value's bit pattern, which is the bit width that bit
    /// packing needs without exceptions.
    pub fn max_bit_width(&self) -> u32 {
        u64::BITS - self.or.leading_zeros()
    }

    /// The number of low bits that are zero in every value, so every value is a multiple of
    /// `2^trailing_zeros`.
    pub fn trailing_zeros(&self) -> u32 {
        self.or.trailing_zeros().min(self.width)
    }

    /// The bits that are the same in every value.
    pub fn constant_bits(&self) -> u64 {
        let type_mask = u64::MAX >> (u64::BITS - self.width);
        (self.and | !self.or) & type_mask
    }
}

impl<T: IntValue> IntAccumulator<T> for CommonBits<T> {
    type Output = CommonBitsResult;

    const USES_FILL: bool = true;

    #[inline(always)]
    fn start(&mut self, _head: T) {}

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        fold_lanes2(
            &mut self.and,
            &mut self.or,
            values,
            |acc, v| acc & v,
            |acc, v| acc | v,
        );
    }

    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], _valid: u64) {
        // Filled nulls repeat valid values, which changes neither the AND nor the OR.
        self.chunk(filled);
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.and[0] = self.and[0] & value;
        self.or[0] = self.or[0] | value;
    }

    #[inline]
    fn finish(self) -> CommonBitsResult {
        CommonBitsResult {
            and: to_bits(self.and.into_iter().fold(!T::zero(), |acc, v| acc & v)),
            or: to_bits(self.or.into_iter().fold(T::zero(), |acc, v| acc | v)),
            width: T::zero().count_zeros(),
        }
    }
}

/// The key of the [`CommonBits`] statistic.
pub struct CommonBitsStat;

impl IntStat for CommonBitsStat {
    type Value = CommonBitsResult;
}

impl<T: IntValue> ErasedAccumulator<T> for CommonBits<T> {
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<CommonBitsStat>(self.finish());
    }
}

/// The number of independent histograms. Consecutive values often have the same bit width, and
/// incrementing one counter repeatedly serializes on store-to-load forwarding.
const HISTOGRAMS: usize = 4;

/// The number of valid values that need each bit width, from `0` to the type's width.
///
/// For 8-bit values, a chunk counts the values below each power of two with vectorized compares,
/// which gives the cumulative histogram: the number of values of at most `w` bits is the number
/// below `2^w`. Wider types need too many compares per value, measured slower from 16 bits, so
/// they increment one counter per value.
#[derive(Debug, Clone)]
pub struct BitWidthHistogram<T> {
    /// Interleaved histograms, indexed by bit width.
    counts: [[u32; 65]; HISTOGRAMS],
    /// For 8-bit values, `below[w]` counts the chunk values below `2^w`.
    below: [u32; 8],
    /// The number of values counted in `below`.
    below_total: u32,
    /// Filled nulls that were counted, by bit width.
    filled: [u32; 65],
    /// The histogrammed type.
    _type: std::marker::PhantomData<T>,
}

impl<T> BitWidthHistogram<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            counts: [[0; 65]; HISTOGRAMS],
            below: [0; 8],
            below_total: 0,
            filled: [0; 65],
            _type: std::marker::PhantomData,
        }
    }
}

impl<T> Default for BitWidthHistogram<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the number of bits needed for `value`'s bit pattern.
#[inline(always)]
fn bit_width<T: PrimInt + AsPrimitive<u64>>(value: T) -> usize {
    // Counting on the zero-extended pattern avoids narrow `leading_zeros`, which is slow for
    // 16-bit values.
    (u64::BITS - to_bits(value).leading_zeros()) as usize
}

/// Adds to `below[w]` the number of `bits` below `2^w`, for each `w` below `B`.
#[inline(always)]
fn count_below<U: PrimInt, const B: usize>(below: &mut [u32; B], bits: &[U; CHUNK]) {
    for (w, count) in below.iter_mut().enumerate() {
        let threshold = U::one() << w;
        // At most 64, so a `u8` sum keeps full-width byte lanes.
        let n: u8 = bits.iter().map(|&b| u8::from(b < threshold)).sum();
        *count += u32::from(n);
    }
}

impl<T: IntValue> IntAccumulator<T> for BitWidthHistogram<T> {
    /// The count of each bit width, from `0` to the type's width.
    type Output = Vec<u32>;

    const USES_FILL: bool = true;

    #[inline(always)]
    fn start(&mut self, _head: T) {}

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        match size_of::<T>() {
            1 => {
                let bits: [u8; CHUNK] = std::array::from_fn(|i| values[i].as_());
                count_below::<u8, 8>(&mut self.below, &bits);
                self.below_total += CHUNK_U32;
            }
            _ => {
                for group in values.as_chunks::<HISTOGRAMS>().0 {
                    for (counts, &v) in self.counts.iter_mut().zip(group) {
                        counts[bit_width(v)] += 1;
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
        self.chunk(filled);
        // Remember the values that filled the nulls, to remove them when finishing.
        for i in null_indices(valid) {
            self.filled[bit_width(filled[i])] += 1;
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.counts[0][bit_width(value)] += 1;
    }

    #[inline]
    fn finish(self) -> Vec<u32> {
        let width = 8 * size_of::<T>();
        let mut histogram: Vec<u32> = (0..=width)
            .map(|w| self.counts.iter().map(|counts| counts[w]).sum())
            .collect();
        if self.below_total > 0 {
            // `below[w]` is the cumulative count of widths up to `w`, and every value has at most
            // `width` bits.
            let cumulative = |w: usize| {
                if w < width {
                    self.below[w]
                } else {
                    self.below_total
                }
            };
            histogram[0] += cumulative(0);
            for w in 1..=width {
                histogram[w] += cumulative(w) - cumulative(w - 1);
            }
        }
        for (count, filled) in histogram.iter_mut().zip(self.filled) {
            *count -= filled;
        }
        histogram
    }
}

/// The key of the [`BitWidthHistogram`] statistic.
pub struct BitWidthHistogramStat;

impl IntStat for BitWidthHistogramStat {
    type Value = Vec<u32>;
}

impl<T: IntValue> ErasedAccumulator<T> for BitWidthHistogram<T> {
    fn finish_into(self, stats: &mut IntStats) {
        stats.insert::<BitWidthHistogramStat>(self.finish());
    }
}
