// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use super::CHUNK;
use super::ChunkAccumulator;
use crate::dtype::NativePType;

/// The sum of the valid integers, in `i64` for signed and `u64` for unsigned integers.
///
/// Integers narrower than 64 bits are summed a chunk or a block at a time, which cannot overflow,
/// and added to the sum with an overflow check. 64-bit integers are checked after every value, as
/// the sum aggregate does, since a signed sum that overflows in between is an overflow even if it
/// later returns into range.
pub struct Sum<T> {
    /// The sum so far of signed integers.
    signed: i64,
    /// The sum so far of unsigned integers.
    unsigned: u64,
    /// Whether the sum overflowed.
    overflow: bool,
    _type: PhantomData<T>,
}

impl<T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>> Default for Sum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>> Sum<T> {
    const SIGNED: bool = T::PTYPE.is_signed_int();

    /// Returns an accumulator that has seen no values.
    pub fn new() -> Self {
        Self {
            signed: 0,
            unsigned: 0,
            overflow: false,
            _type: PhantomData,
        }
    }

    /// Returns the sum, or `None` if it overflowed.
    pub fn finish(&self) -> Option<i128> {
        (!self.overflow).then(|| {
            if Self::SIGNED {
                i128::from(self.signed)
            } else {
                i128::from(self.unsigned)
            }
        })
    }

    /// Adds a value, or the sum of values narrower than 64 bits, widened to 64 bits.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, signed: i64, unsigned: u64) {
        let overflow = if Self::SIGNED {
            let (sum, overflow) = self.signed.overflowing_add(signed);
            self.signed = sum;
            overflow
        } else {
            let (sum, overflow) = self.unsigned.overflowing_add(unsigned);
            self.unsigned = sum;
            overflow
        };
        self.overflow |= overflow;
    }

    /// Adds valid values.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn values(&mut self, values: &[T]) {
        if size_of::<T>() < 8 {
            // At most a block of values of at most 32 bits, so neither sum can overflow.
            if Self::SIGNED {
                self.add(values.iter().map(|&v| AsPrimitive::<i64>::as_(v)).sum(), 0);
            } else {
                self.add(0, values.iter().map(|&v| AsPrimitive::<u64>::as_(v)).sum());
            }
        } else {
            for &v in values {
                self.add(v.as_(), v.as_());
            }
        }
    }
}

impl<T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>> ChunkAccumulator<T> for Sum<T> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.values(values);
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        self.values(chunks.as_flattened());
    }

    fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, _len: usize) {
        let mut valid = valid;
        while valid != 0 {
            let value = values[valid.trailing_zeros() as usize];
            self.add(value.as_(), value.as_());
            valid &= valid - 1;
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn is_done(&self) -> bool {
        self.overflow
    }
}
