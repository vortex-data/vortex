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
use vortex_buffer::{Buffer, BufferMut};

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
    type ALPInt: PrimInt + Display + ToPrimitive + Copy;

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
                .take(SAMPLE_SIZE)
                .cloned()
                .collect_vec()
        });

        for e in (0..Self::MAX_EXPONENT).rev() {
            for f in 0..e {
                let (encoded, exceptional_positions) =
                    Self::encode(sample.as_deref().unwrap_or(values), Exponents { e, f });

                let size = Self::estimate_encoded_size(&encoded, exceptional_positions.len());
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
    fn estimate_encoded_size(encoded: &[Self::ALPInt], n_exceptions: usize) -> usize {
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
        let patch_bytes = n_exceptions * (size_of::<Self>() + size_of::<u16>());

        encoded_bytes + patch_bytes
    }

    /// A quantity of [Self] expected to fit into L1 cache.
    const ENCODE_CHUNK_SIZE: usize = (32 << 10) / size_of::<Self::ALPInt>();

    /// ALP encode chunk-by-chunk.
    ///
    /// Unlike [Self::encode], this operation processes no more than [Self::ENCODE_CHUNK_SIZE]
    /// elements at once which can make better use of the L1 cache because [Self::encode] makes two
    /// passes over `values`: first to encode and second to extract the exceptional values.
    fn encode_chunkwise(
        values: &[Self],
        exponents: Exponents,
    ) -> (Buffer<Self::ALPInt>, Buffer<u64>) {
        let mut encoded = BufferMut::<Self::ALPInt>::with_capacity(values.len());
        let mut patch_indices = BufferMut::<u64>::empty();
        for chunk in values.chunks(Self::ENCODE_CHUNK_SIZE) {
            let (encoded_chunk, patches_indices_chunk) = Self::encode(chunk, exponents);
            encoded.extend(encoded_chunk);
            patch_indices.extend(patches_indices_chunk);
        }
        (encoded.freeze(), patch_indices.freeze())
    }

    /// ALP encode the given values using the given exponents.
    ///
    /// The index of each value for which encode-decode is not the identity function is returned.
    ///
    /// See also: [Self::encode_chunkwise].
    #[allow(clippy::cast_possible_truncation)] // The patch_indices are known to be valid indices into encoded.
    fn encode(values: &[Self], exponents: Exponents) -> (Vec<Self::ALPInt>, Vec<u64>) {
        let (mut encoded, needs_patch): (Vec<Self::ALPInt>, Vec<bool>) = values
            .iter()
            .map(|value| {
                let encoded = unsafe { Self::encode_single_unchecked(*value, exponents) };
                let maybe_decoded = Self::decode_single(encoded, exponents);
                let needs_patch = maybe_decoded != *value;
                (encoded, needs_patch)
            })
            .unzip();

        // Patched values either have tiny differences (e.g. 1.01, 1.02, 1.00000001) or big
        // differences (e.g. 1.01, 1.02, 1000.0). In the latter case, this large value
        // prevents bitpacking (or forces patches into bitpacking).
        //
        // Zero allows bitpacking but prevents frame-of-reference encoding, so we choose the first
        // successfully encoded value.
        let patch_indices: Vec<u64> = needs_patch
            .into_iter()
            .enumerate()
            .filter(|(_, needs_patch)| *needs_patch)
            .map(|(index, _)| index as u64)
            .collect();

        if let Some(fill_value) =
            Self::find_first_non_patched_encoded_value(&encoded, &patch_indices)
        {
            for index in patch_indices.iter() {
                let index = *index as usize;
                encoded[index] = fill_value;
            }
        }

        (encoded, patch_indices)
    }

    fn find_first_non_patched_encoded_value(
        encoded: &[Self::ALPInt],
        patch_indices: &[u64],
    ) -> Option<Self::ALPInt> {
        for index in 0..encoded.len() {
            if index >= patch_indices.len() || patch_indices[index] != index as u64 {
                return Some(encoded[index]);
            }
        }
        None
    }

    #[inline]
    fn encode_single(value: Self, exponents: Exponents) -> Option<Self::ALPInt> {
        let encoded = unsafe { Self::encode_single_unchecked(value, exponents) };
        let decoded = Self::decode_single(encoded, exponents);
        if decoded == value {
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
        encoded.map_each(move |encoded| Self::decode_single(encoded, exponents))
    }

    #[inline(always)]
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
