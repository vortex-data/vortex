// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::mem::size_of;
use std::mem::transmute;
use std::mem::transmute_copy;

use itertools::Itertools;
use num_traits::CheckedSub;
use num_traits::Float;
use num_traits::PrimInt;
use num_traits::ToPrimitive;

mod array;
mod compress;
mod compute;
mod decompress;
mod ops;
mod rules;

#[cfg(test)]
mod tests {
    use vortex_array::ProstMetadata;
    use vortex_array::dtype::PType;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;

    use crate::alp::array::ALPMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alp_metadata() {
        check_metadata(
            "alp.metadata",
            ProstMetadata(ALPMetadata {
                patches: Some(PatchesMetadata::new(
                    usize::MAX,
                    usize::MAX,
                    PType::U64,
                    None,
                    None,
                    None,
                )),
                exp_e: u32::MAX,
                exp_f: u32::MAX,
            }),
        );
    }
}

pub use array::*;
pub use compress::alp_encode;
pub use decompress::decompress_into_array;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::PValue;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

const SAMPLE_SIZE: usize = 32;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Exponents {
    pub e: u8,
    pub f: u8,
}

impl Display for Exponents {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "e: {}, f: {}", self.e, self.f)
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

pub trait ALPFloat: private::Sealed + Float + Display + NativePType {
    type ALPInt: PrimInt + Display + ToPrimitive + Copy + NativePType + Into<PValue>;

    const FRACTIONAL_BITS: u8;
    const MAX_EXPONENT: u8;
    const SWEET: Self;
    const F10: &'static [Self];
    const IF10: &'static [Self];

    /// Round to the nearest floating integer by shifting in and out of the low precision range.
    #[inline]
    fn fast_round(self) -> Self {
        (self + Self::SWEET) - Self::SWEET
    }

    /// Equivalent to calling `as` to cast the primitive float to the target integer type.
    fn as_int(self) -> Self::ALPInt;

    /// Convert from the integer type back to the float type using `as`.
    fn from_int(n: Self::ALPInt) -> Self;

    fn find_best_exponents(values: &[Self]) -> Exponents {
        let mut best_exp = Exponents { e: 0, f: 0 };
        let mut best_nbytes: usize = usize::MAX;

        let sample = (values.len() > SAMPLE_SIZE).then(|| {
            values
                .iter()
                .step_by(values.len() / SAMPLE_SIZE)
                .cloned()
                .collect_vec()
        });
        let sample_values = sample.as_deref().unwrap_or(values);

        for e in (0..Self::MAX_EXPONENT).rev() {
            for f in 0..e {
                let exp = Exponents { e, f };
                let size = Self::estimate_for_exponents(sample_values, exp);
                if size < best_nbytes {
                    best_nbytes = size;
                    best_exp = exp;
                } else if size == best_nbytes && e - f < best_exp.e - best_exp.f {
                    best_exp = exp;
                }
            }

            // If the best exponents so far produced zero patches on the sample,
            // that is likely optimal -- no need to search lower e values.
            if best_nbytes > 0 && best_exp.e == e {
                let patch_free = sample_values.iter().all(|&v| {
                    let encoded = Self::encode_single_unchecked(v, best_exp);
                    Self::decode_single(encoded, best_exp).is_eq(v)
                });
                if patch_free {
                    break;
                }
            }
        }

        best_exp
    }

    /// Lightweight estimation of encoded size for a given set of exponents.
    ///
    /// Unlike [`Self::encode`], this avoids allocating output buffers, tracking
    /// chunk offsets, gathering patch values, and computing fill values. It only
    /// counts mismatches (patches) and tracks the encoded value range to estimate
    /// bitwidth.
    #[inline]
    fn estimate_for_exponents(sample: &[Self], exponents: Exponents) -> usize {
        let mut patch_count: usize = 0;
        let mut enc_min: Option<Self::ALPInt> = None;
        let mut enc_max: Option<Self::ALPInt> = None;

        for &v in sample {
            let encoded = Self::encode_single_unchecked(v, exponents);
            let decoded = Self::decode_single(encoded, exponents);
            if !decoded.is_eq(v) {
                patch_count += 1;
            } else {
                enc_min = Some(match enc_min {
                    Some(cur) if cur < encoded => cur,
                    _ => encoded,
                });
                enc_max = Some(match enc_max {
                    Some(cur) if cur > encoded => cur,
                    _ => encoded,
                });
            }
        }

        let bits_per_encoded = enc_min
            .zip(enc_max)
            .and_then(|(min, max)| max.checked_sub(&min))
            .and_then(|range| range.to_u64())
            .and_then(|range| {
                range
                    .checked_ilog2()
                    .map(|bits| (bits + 1) as usize)
                    .or(Some(0))
            })
            .unwrap_or(size_of::<Self::ALPInt>() * 8);

        let encoded_bytes = (sample.len() * bits_per_encoded).div_ceil(8);
        let patch_bytes = patch_count * (size_of::<Self>() + size_of::<u16>());

        encoded_bytes + patch_bytes
    }

    #[inline]
    fn estimate_encoded_size(encoded: &[Self::ALPInt], patches: &[Self]) -> usize {
        let bits_per_encoded = encoded
            .iter()
            .minmax()
            .into_option()
            // estimating bits per encoded value assuming frame-of-reference + bitpacking-without-patches
            .and_then(|(min, max)| max.checked_sub(min))
            .and_then(|range_size: <Self as ALPFloat>::ALPInt| range_size.to_u64())
            .and_then(|range_size| {
                range_size
                    .checked_ilog2()
                    .map(|bits| (bits + 1) as usize)
                    .or(Some(0))
            })
            .unwrap_or(size_of::<Self::ALPInt>() * 8);

        let encoded_bytes = (encoded.len() * bits_per_encoded).div_ceil(8);
        // each patch is a value + a position
        // in practice, patch positions are in [0, u16::MAX] because of how we chunk
        let patch_bytes = patches.len() * (size_of::<Self>() + size_of::<u16>());

        encoded_bytes + patch_bytes
    }

    #[expect(
        clippy::type_complexity,
        reason = "tuple return type is appropriate for multiple encoding outputs"
    )]
    fn encode(
        values: &[Self],
        exponents: Option<Exponents>,
    ) -> (
        Exponents,
        Buffer<Self::ALPInt>,
        Buffer<u64>,
        Buffer<Self>,
        BufferMut<u64>,
    ) {
        let exp = exponents.unwrap_or_else(|| Self::find_best_exponents(values));

        let mut encoded_output = BufferMut::<Self::ALPInt>::with_capacity(values.len());

        // Estimate capacity to be one patch per 32 values.
        let mut patch_indices = BufferMut::<u64>::with_capacity(values.len() / 32);
        let mut patch_values = BufferMut::<Self>::with_capacity(values.len() / 32);

        // There's exactly one offset per 1024 chunk.
        let mut chunk_offsets = BufferMut::<u64>::with_capacity(values.len().div_ceil(1024));
        let mut fill_value: Option<Self::ALPInt> = None;

        for chunk in values.chunks(1024) {
            chunk_offsets.push(patch_indices.len() as u64);
            encode_chunk_unchecked(
                chunk,
                exp,
                &mut encoded_output,
                &mut patch_indices,
                &mut patch_values,
                &mut fill_value,
            );
        }

        (
            exp,
            encoded_output.freeze(),
            patch_indices.freeze(),
            patch_values.freeze(),
            chunk_offsets,
        )
    }

    #[inline]
    fn encode_single(value: Self, exponents: Exponents) -> Option<Self::ALPInt> {
        let encoded = Self::encode_single_unchecked(value, exponents);
        let decoded = Self::decode_single(encoded, exponents);
        if decoded.is_eq(value) {
            return Some(encoded);
        }
        None
    }

    fn encode_above(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .ceil()
            .as_int()
    }

    fn encode_below(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .floor()
            .as_int()
    }

    fn decode(encoded: &[Self::ALPInt], exponents: Exponents) -> Vec<Self> {
        let mut values = Vec::with_capacity(encoded.len());
        for encoded in encoded {
            values.push(Self::decode_single(*encoded, exponents));
        }
        values
    }

    fn decode_buffer(encoded: BufferMut<Self::ALPInt>, exponents: Exponents) -> BufferMut<Self> {
        let factor_f = Self::F10[exponents.f as usize];
        let factor_e = Self::IF10[exponents.e as usize];
        encoded.map_each_in_place(move |encoded| Self::from_int(encoded) * factor_f * factor_e)
    }

    fn decode_into(encoded: &[Self::ALPInt], exponents: Exponents, output: &mut [Self]) {
        assert_eq!(encoded.len(), output.len());
        let factor_f = Self::F10[exponents.f as usize];
        let factor_e = Self::IF10[exponents.e as usize];
        for i in 0..encoded.len() {
            output[i] = Self::from_int(encoded[i]) * factor_f * factor_e;
        }
    }

    fn decode_slice_inplace(encoded: &mut [Self::ALPInt], exponents: Exponents) {
        let factor_f = Self::F10[exponents.f as usize];
        let factor_e = Self::IF10[exponents.e as usize];
        let decoded: &mut [Self] = unsafe { transmute(encoded) };
        for v in decoded.iter_mut() {
            let int_val: Self::ALPInt = unsafe { transmute_copy(v) };
            *v = Self::from_int(int_val) * factor_f * factor_e;
        }
    }

    #[inline(always)]
    fn decode_single(encoded: Self::ALPInt, exponents: Exponents) -> Self {
        Self::from_int(encoded) * Self::F10[exponents.f as usize] * Self::IF10[exponents.e as usize]
    }

    /// Encode single float value. The returned value might decode to a different value than passed in.
    /// Consider using [`Self::encode_single`] if you want the checked version of this function.
    #[inline(always)]
    fn encode_single_unchecked(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .fast_round()
            .as_int()
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional truncation for ALP encoding"
)]
fn encode_chunk_unchecked<T: ALPFloat>(
    chunk: &[T],
    exp: Exponents,
    encoded_output: &mut BufferMut<T::ALPInt>,
    patch_indices: &mut BufferMut<u64>,
    patch_values: &mut BufferMut<T>,
    fill_value: &mut Option<T::ALPInt>,
) {
    let num_prev_encoded = encoded_output.len();
    let num_prev_patches = patch_indices.len();
    assert_eq!(patch_indices.len(), patch_values.len());
    let has_filled = fill_value.is_some();

    // Pre-compute factors to avoid repeated table lookups in the hot loop.
    let factor_e = T::F10[exp.e as usize];
    let factor_if = T::IF10[exp.f as usize];
    let factor_f = T::F10[exp.f as usize];
    let factor_ie = T::IF10[exp.e as usize];

    // Encode all values in a single pass with pre-computed factors.
    encoded_output.extend_trusted(
        chunk
            .iter()
            .map(|&v| (v * factor_e * factor_if).fast_round().as_int()),
    );
    assert_eq!(encoded_output.len(), num_prev_encoded + chunk.len());

    // Find patches by checking decode equality, collecting indices and values directly.
    // This eliminates the previous two-pass approach that first counted patches and then
    // re-decoded every value to gather them.
    let encoded_slice = &encoded_output[num_prev_encoded..];
    for (i, (&original, &encoded)) in chunk.iter().zip(encoded_slice.iter()).enumerate() {
        let decoded = T::from_int(encoded) * factor_f * factor_ie;
        if !decoded.is_eq(original) {
            patch_indices.push((num_prev_encoded + i) as u64);
            patch_values.push(original);
        }
    }
    let chunk_patch_count = patch_indices.len() - num_prev_patches;

    // find the first successfully encoded value (i.e., not patched)
    // this is our fill value for missing values
    if fill_value.is_none() && (chunk_patch_count < chunk.len()) {
        assert_eq!(num_prev_encoded, num_prev_patches);
        for i in num_prev_encoded..encoded_output.len() {
            if i >= patch_indices.len() || patch_indices[i] != i as u64 {
                *fill_value = Some(encoded_output[i]);
                break;
            }
        }
    }

    // replace the patched values in the encoded array with the fill value
    // for better downstream compression
    if let Some(fill_value) = fill_value {
        // handle the edge case where the first N >= 1 chunks are all patches
        let start_patch = if !has_filled { 0 } else { num_prev_patches };
        for patch_idx in &patch_indices[start_patch..] {
            encoded_output[*patch_idx as usize] = *fill_value;
        }
    }
}

impl ALPFloat for f32 {
    type ALPInt = i32;
    const FRACTIONAL_BITS: u8 = 23;
    const MAX_EXPONENT: u8 = 10;
    const SWEET: Self =
        (1 << Self::FRACTIONAL_BITS) as Self + (1 << (Self::FRACTIONAL_BITS - 1)) as Self;

    const F10: &'static [Self] = &[
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0, // 10^10
    ];
    const IF10: &'static [Self] = &[
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001, // 10^-10
    ];

    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional float to int truncation for ALP encoding"
    )]
    fn as_int(self) -> Self::ALPInt {
        self as _
    }

    #[inline(always)]
    fn from_int(n: Self::ALPInt) -> Self {
        n as _
    }
}

impl ALPFloat for f64 {
    type ALPInt = i64;
    const FRACTIONAL_BITS: u8 = 52;
    const MAX_EXPONENT: u8 = 18; // 10^18 is the maximum i64
    const SWEET: Self =
        (1u64 << Self::FRACTIONAL_BITS) as Self + (1u64 << (Self::FRACTIONAL_BITS - 1)) as Self;
    const F10: &'static [Self] = &[
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0,
        100000000000.0,
        1000000000000.0,
        10000000000000.0,
        100000000000000.0,
        1000000000000000.0,
        10000000000000000.0,
        100000000000000000.0,
        1000000000000000000.0,
        10000000000000000000.0,
        100000000000000000000.0,
        1000000000000000000000.0,
        10000000000000000000000.0,
        100000000000000000000000.0, // 10^23
    ];

    const IF10: &'static [Self] = &[
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001,
        0.00000000001,
        0.000000000001,
        0.0000000000001,
        0.00000000000001,
        0.000000000000001,
        0.0000000000000001,
        0.00000000000000001,
        0.000000000000000001,
        0.0000000000000000001,
        0.00000000000000000001,
        0.000000000000000000001,
        0.0000000000000000000001,
        0.00000000000000000000001, // 10^-23
    ];

    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional float to int truncation for ALP encoding"
    )]
    fn as_int(self) -> Self::ALPInt {
        self as _
    }

    #[inline(always)]
    fn from_int(n: Self::ALPInt) -> Self {
        n as _
    }
}
