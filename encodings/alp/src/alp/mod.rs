use std::fmt::{Display, Formatter};
use std::mem::size_of;

use itertools::Itertools;
use num_traits::{CheckedSub, Float, PrimInt, ToPrimitive};
use serde::{Deserialize, Serialize};

mod array;
mod compress;
mod compute;

pub use array::*;
pub use compress::*;

const SAMPLE_SIZE: usize = 32;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

pub trait ALPFloat: private::Sealed + Float + Display + 'static {
    type ALPInt: PrimInt + Display + ToPrimitive;

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

        for e in (0..Self::MAX_EXPONENT).rev() {
            for f in 0..e {
                let (_, encoded, _, exc_patches) = Self::encode(
                    sample.as_deref().unwrap_or(values),
                    Some(Exponents { e, f }),
                );

                let size = Self::estimate_encoded_size(&encoded, &exc_patches);
                if size < best_nbytes {
                    best_nbytes = size;
                    best_exp = Exponents { e, f };
                } else if size == best_nbytes && e - f < best_exp.e - best_exp.f {
                    best_exp = Exponents { e, f };
                }
            }
        }

        best_exp
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

        let encoded_bytes = (encoded.len() * bits_per_encoded + 7) / 8;
        // each patch is a value + a position
        // in practice, patch positions are in [0, u16::MAX] because of how we chunk
        let patch_bytes = patches.len() * (size_of::<Self>() + size_of::<u16>());

        encoded_bytes + patch_bytes
    }

    fn encode(
        values: &[Self],
        exponents: Option<Exponents>,
    ) -> (Exponents, Vec<Self::ALPInt>, Vec<u64>, Vec<Self>) {
        let exp = exponents.unwrap_or_else(|| Self::find_best_exponents(values));

        let mut encoded_output = Vec::with_capacity(values.len());
        let mut patch_indices = Vec::new();
        let mut patch_values = Vec::new();
        let mut fill_value: Option<Self::ALPInt> = None;

        // this is intentionally branchless
        // we batch this into 32KB of values at a time to make it more L1 cache friendly
        let encode_chunk_size: usize = (32 << 10) / size_of::<Self::ALPInt>();
        for chunk in values.chunks(encode_chunk_size) {
            encode_chunk_unchecked(
                chunk,
                exp,
                &mut encoded_output,
                &mut patch_indices,
                &mut patch_values,
                &mut fill_value,
            );
        }

        (exp, encoded_output, patch_indices, patch_values)
    }

    #[inline]
    fn encode_single(value: Self, exponents: Exponents) -> Result<Self::ALPInt, Self> {
        let encoded = unsafe { Self::encode_single_unchecked(value, exponents) };
        let decoded = Self::decode_single(encoded, exponents);
        if decoded == value {
            return Ok(encoded);
        }
        Err(value)
    }

    fn decode(encoded: &[Self::ALPInt], exponents: Exponents) -> Vec<Self> {
        let mut values = Vec::with_capacity(encoded.len());
        for encoded in encoded {
            values.push(Self::decode_single(*encoded, exponents));
        }
        values
    }

    #[inline]
    fn decode_single(encoded: Self::ALPInt, exponents: Exponents) -> Self {
        Self::from_int(encoded) * Self::F10[exponents.f as usize] * Self::IF10[exponents.e as usize]
    }

    /// # Safety
    ///
    /// The returned value may not decode back to the original value.
    #[inline(always)]
    unsafe fn encode_single_unchecked(value: Self, exponents: Exponents) -> Self::ALPInt {
        (value * Self::F10[exponents.e as usize] * Self::IF10[exponents.f as usize])
            .fast_round()
            .as_int()
    }
}

#[allow(clippy::cast_possible_truncation)]
fn encode_chunk_unchecked<T: ALPFloat>(
    chunk: &[T],
    exp: Exponents,
    encoded_output: &mut Vec<T::ALPInt>,
    patch_indices: &mut Vec<u64>,
    patch_values: &mut Vec<T>,
    fill_value: &mut Option<T::ALPInt>,
) {
    let num_prev_encoded = encoded_output.len();
    let num_prev_patches = patch_indices.len();
    assert_eq!(patch_indices.len(), patch_values.len());
    let has_filled = fill_value.is_some();

    // encode the chunk, counting the number of patches
    let mut chunk_patch_count = 0;
    encoded_output.extend(chunk.iter().map(|v| {
        let encoded = unsafe { T::encode_single_unchecked(*v, exp) };
        let decoded = T::decode_single(encoded, exp);
        let neq = (decoded != *v) as usize;
        chunk_patch_count += neq;
        encoded
    }));
    let chunk_patch_count = chunk_patch_count; // immutable hereafter
    assert_eq!(encoded_output.len(), num_prev_encoded + chunk.len());

    if chunk_patch_count > 0 {
        // we need to gather the patches for this chunk
        // preallocate space for the patches (plus one because our loop may attempt to write one past the end)
        patch_indices.reserve(chunk_patch_count + 1);
        patch_values.reserve(chunk_patch_count + 1);

        // record the patches in this chunk
        let patch_indices_mut = patch_indices.spare_capacity_mut();
        let patch_values_mut = patch_values.spare_capacity_mut();
        let mut chunk_patch_index = 0;
        for i in num_prev_encoded..encoded_output.len() {
            let decoded = T::decode_single(encoded_output[i], exp);
            // write() is only safe to call more than once because the values are primitive (i.e., Drop is a no-op)
            patch_indices_mut[chunk_patch_index].write(i as u64);
            patch_values_mut[chunk_patch_index].write(chunk[i - num_prev_encoded]);
            chunk_patch_index += (decoded != chunk[i - num_prev_encoded]) as usize;
        }
        assert_eq!(chunk_patch_index, chunk_patch_count);
        unsafe {
            patch_indices.set_len(num_prev_patches + chunk_patch_count);
            patch_values.set_len(num_prev_patches + chunk_patch_count);
        }
    }

    // find the first successfully encoded value (i.e., not patched)
    // this is our fill value for missing values
    if fill_value.is_none() && (num_prev_encoded + chunk_patch_count < encoded_output.len()) {
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
    #[allow(clippy::cast_possible_truncation)]
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
    #[allow(clippy::cast_possible_truncation)]
    fn as_int(self) -> Self::ALPInt {
        self as _
    }

    #[inline(always)]
    fn from_int(n: Self::ALPInt) -> Self {
        n as _
    }
}
