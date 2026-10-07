// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Floats as integer keys, so that the integer accumulators order and compare them.

use std::marker::PhantomData;

use super::BLOCK;
use super::CHUNK;
use super::ChunkAccumulator;
use super::Extremes;
use crate::dtype::NativePType;
use crate::dtype::half::f16;

/// A float with an integer key whose order is [`f64::total_cmp`]'s, and whose equality is the
/// equality of bits, as [`NativePType::total_compare`] and [`NativePType::is_eq`] define them.
pub trait FloatKey: NativePType {
    /// The signed integer of the same width.
    type Key: Extremes;

    /// The key of `self`. The mapping is its own inverse.
    fn key(self) -> Self::Key;

    /// The float whose key is `key`.
    fn from_key(key: Self::Key) -> Self;

    /// Whether `key` is the key of a NaN.
    fn key_is_nan(key: Self::Key) -> bool;
}

/// Implements [`FloatKey`] as [`f64::total_cmp`] orders bits: negative values have every bit but
/// the sign flipped, so that their order reverses.
macro_rules! impl_float_key {
    ($F:ty, $Key:ty, $Bits:ty, $to_bits:expr, $from_bits:expr) => {
        impl FloatKey for $F {
            type Key = $Key;

            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn key(self) -> $Key {
                let bits = $to_bits(self) as $Key;
                bits ^ ((((bits >> (<$Key>::BITS - 1)) as $Bits) >> 1) as $Key)
            }

            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn from_key(key: $Key) -> Self {
                let bits = key ^ ((((key >> (<$Key>::BITS - 1)) as $Bits) >> 1) as $Key);
                $from_bits(bits as $Bits)
            }

            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn key_is_nan(key: $Key) -> bool {
                // NaNs order beyond the infinities of their sign.
                key > <$F>::INFINITY.key() || key < <$F>::NEG_INFINITY.key()
            }
        }
    };
}

impl_float_key!(f16, i16, u16, f16::to_bits, f16::from_bits);
impl_float_key!(f32, i32, u32, f32::to_bits, f32::from_bits);
impl_float_key!(f64, i64, u64, f64::to_bits, f64::from_bits);

/// Returns the keys of a chunk.
#[allow(clippy::inline_always)]
#[inline(always)]
fn keys<F: FloatKey>(values: &[F; CHUNK]) -> [F::Key; CHUNK] {
    std::array::from_fn(|i| values[i].key())
}

/// Returns the word whose bits are set for the NaN keys.
#[allow(clippy::inline_always)]
#[inline(always)]
fn nans<F: FloatKey>(keys: &[F::Key; CHUNK]) -> u64 {
    // Branch-free, so that the common check that no key is NaN vectorizes.
    if !keys.iter().fold(false, |any, &k| any | F::key_is_nan(k)) {
        return 0;
    }
    keys.iter()
        .enumerate()
        .fold(0, |word, (i, &k)| word | (u64::from(F::key_is_nan(k)) << i))
}

/// An integer accumulator `A` of float keys: the float values are ordered by
/// [`NativePType::total_compare`] and equal when their bits are. NaNs are nulls if `SKIP_NANS`.
pub struct Keyed<F: FloatKey, A, const SKIP_NANS: bool> {
    inner: A,
    _type: PhantomData<F>,
}

impl<F: FloatKey, A, const SKIP_NANS: bool> Keyed<F, A, SKIP_NANS> {
    /// Wraps an accumulator of keys.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            _type: PhantomData,
        }
    }

    /// Returns the accumulator of keys.
    pub fn inner(&self) -> &A {
        &self.inner
    }
}

impl<F, A, const SKIP_NANS: bool> ChunkAccumulator<F> for Keyed<F, A, SKIP_NANS>
where
    F: FloatKey,
    A: ChunkAccumulator<F::Key>,
{
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[F; CHUNK]) {
        let keys = keys(values);
        let nans = if SKIP_NANS { nans::<F>(&keys) } else { 0 };
        if nans == 0 {
            self.inner.chunk(&keys);
        } else {
            self.inner.partial_chunk(&keys, !nans, CHUNK);
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn block(&mut self, chunks: &[[F; CHUNK]]) {
        let mut keyed = [[F::Key::LOWEST; CHUNK]; BLOCK];
        for (keys, chunk) in keyed.iter_mut().zip(chunks) {
            *keys = self::keys(chunk);
        }
        let keyed = &keyed[..chunks.len()];
        if SKIP_NANS && keyed.iter().any(|keys| nans::<F>(keys) != 0) {
            for keys in keyed {
                let nans = nans::<F>(keys);
                self.inner.partial_chunk(keys, !nans, CHUNK);
            }
        } else {
            self.inner.block(keyed);
        }
    }

    fn partial_chunk(&mut self, values: &[F; CHUNK], valid: u64, len: usize) {
        let keys = keys(values);
        let valid = if SKIP_NANS {
            valid & !nans::<F>(&keys)
        } else {
            valid
        };
        self.inner.partial_chunk(&keys, valid, len);
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }
}

/// The sum of the valid floats in `f64`, skipping NaNs if `SKIP_NANS` and otherwise turning NaN.
///
/// The values are added one at a time in order, as the sum aggregate adds them, so that the
/// rounding is the same.
pub struct FloatSum<F, const SKIP_NANS: bool> {
    sum: f64,
    _type: PhantomData<F>,
}

impl<F: NativePType, const SKIP_NANS: bool> Default for FloatSum<F, SKIP_NANS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: NativePType, const SKIP_NANS: bool> FloatSum<F, SKIP_NANS> {
    /// Returns an accumulator that has seen no values.
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            _type: PhantomData,
        }
    }

    /// Returns the sum.
    pub fn finish(&self) -> f64 {
        self.sum
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, value: F) {
        if !(SKIP_NANS && value.is_nan()) {
            self.sum += value.to_f64().unwrap_or(f64::NAN);
        }
    }
}

impl<F: NativePType, const SKIP_NANS: bool> ChunkAccumulator<F> for FloatSum<F, SKIP_NANS> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[F; CHUNK]) {
        for &value in values {
            self.add(value);
        }
    }

    fn partial_chunk(&mut self, values: &[F; CHUNK], valid: u64, _len: usize) {
        let mut valid = valid;
        while valid != 0 {
            self.add(values[valid.trailing_zeros() as usize]);
            valid &= valid - 1;
        }
    }
}
