#![allow(clippy::cast_possible_truncation)]

pub use array::*;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;

mod array;
mod compute;
mod serde;
mod variants;

use std::ops::{Shl, Shr};

use itertools::Itertools;
use num_traits::{Float, One, PrimInt};
use rustc_hash::FxBuildHasher;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::{Array, IntoArray, ToCanonical};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail};
use vortex_fastlanes::bitpack_encode_unchecked;

use crate::match_each_alp_float_ptype;

macro_rules! bit_width {
    ($value:expr) => {
        if $value == 0 {
            1
        } else {
            $value.ilog2().wrapping_add(1) as usize
        }
    };
}

/// Max number of bits to cut from the MSB section of each float.
const CUT_LIMIT: usize = 16;

const MAX_DICT_SIZE: u8 = 8;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Main trait for ALP-RD encodable floating point numbers.
///
/// Like the paper, we limit this to the IEEE7 754 single-precision (`f32`) and double-precision
/// (`f64`) floating point types.
pub trait ALPRDFloat: private::Sealed + Float + Copy + NativePType {
    /// The unsigned integer type with the same bit-width as the floating-point type.
    type UINT: NativePType + PrimInt + One + Copy;

    /// Number of bits the value occupies in registers.
    const BITS: usize = size_of::<Self>() * 8;

    /// Bit-wise transmute from the unsigned integer type to the floating-point type.
    fn from_bits(bits: Self::UINT) -> Self;

    /// Bit-wise transmute into the unsigned integer type.
    fn to_bits(value: Self) -> Self::UINT;

    /// Truncating conversion from the unsigned integer type to `u16`.
    fn to_u16(bits: Self::UINT) -> u16;

    /// Type-widening conversion from `u16` to the unsigned integer type.
    fn from_u16(value: u16) -> Self::UINT;
}

impl ALPRDFloat for f64 {
    type UINT = u64;

    fn from_bits(bits: Self::UINT) -> Self {
        f64::from_bits(bits)
    }

    fn to_bits(value: Self) -> Self::UINT {
        value.to_bits()
    }

    fn to_u16(bits: Self::UINT) -> u16 {
        bits as u16
    }

    fn from_u16(value: u16) -> Self::UINT {
        value as u64
    }
}

impl ALPRDFloat for f32 {
    type UINT = u32;

    fn from_bits(bits: Self::UINT) -> Self {
        f32::from_bits(bits)
    }

    fn to_bits(value: Self) -> Self::UINT {
        value.to_bits()
    }

    fn to_u16(bits: Self::UINT) -> u16 {
        bits as u16
    }

    fn from_u16(value: u16) -> Self::UINT {
        value as u32
    }
}

/// Encoder for ALP-RD ("real doubles") values.
///
/// The encoder calculates its parameters from a single sample of floating-point values,
/// and then can be applied to many vectors.
///
/// ALP-RD uses the algorithm outlined in Section 3.4 of the paper. The crux of it is that the front
/// (most significant) bits of many double vectors tend to be  the same, i.e. most doubles in a
/// vector often use the same exponent and front bits. Compression proceeds by finding the best
/// prefix of up to 16 bits that can be collapsed into a dictionary of
/// up to 8 elements. Each double can then be broken into the front/left `L` bits, which neatly
/// bit-packs down to 1-3 bits per element (depending on the actual dictionary size).
/// The remaining `R` bits naturally bit-pack.
///
/// In the ideal case, this scheme allows us to store a sequence of doubles in 49 bits-per-value.
///
/// Our implementation draws on the MIT-licensed [C++ implementation] provided by the original authors.
///
/// [C++ implementation]: https://github.com/cwida/ALP/blob/main/include/alp/rd.hpp
pub struct RDEncoder {
    right_bit_width: u8,
    codes: Vec<u16>,
}

impl RDEncoder {
    /// Build a new encoder from a sample of doubles.
    pub fn new<T>(sample: &[T]) -> Self
    where
        T: ALPRDFloat + NativePType,
        T::UINT: NativePType,
    {
        let dictionary = find_best_dictionary::<T>(sample);

        let mut codes = vec![0; dictionary.dictionary.len()];
        dictionary.dictionary.into_iter().for_each(|(bits, code)| {
            // write the reverse mapping into the codes vector.
            codes[code as usize] = bits
        });

        Self {
            right_bit_width: dictionary.right_bit_width,
            codes,
        }
    }

    /// Encode a set of floating point values with ALP-RD.
    ///
    /// Each value will be split into a left and right component, which are compressed individually.
    pub fn encode(&self, array: &PrimitiveArray) -> ALPRDArray {
        match_each_alp_float_ptype!(array.ptype(), |$P| {
            self.encode_generic::<$P>(array)
        })
    }

    fn encode_generic<T>(&self, array: &PrimitiveArray) -> ALPRDArray
    where
        T: ALPRDFloat + NativePType,
        T::UINT: NativePType,
    {
        assert!(
            !self.codes.is_empty(),
            "codes lookup table must be populated before RD encoding"
        );

        let doubles = array.as_slice::<T>();

        let mut left_parts: BufferMut<u16> = BufferMut::with_capacity(doubles.len());
        let mut right_parts: BufferMut<T::UINT> = BufferMut::with_capacity(doubles.len());
        let mut exceptions_pos: BufferMut<u64> = BufferMut::with_capacity(doubles.len() / 4);
        let mut exceptions: BufferMut<u16> = BufferMut::with_capacity(doubles.len() / 4);

        // mask for right-parts
        let right_mask = T::UINT::one().shl(self.right_bit_width as _) - T::UINT::one();
        let max_code = self.codes.len() - 1;
        let left_bit_width = bit_width!(max_code);

        for v in doubles.iter().copied() {
            right_parts.push(T::to_bits(v) & right_mask);
            left_parts.push(<T as ALPRDFloat>::to_u16(
                T::to_bits(v).shr(self.right_bit_width as _),
            ));
        }

        // dict-encode the left-parts, keeping track of exceptions
        for (idx, left) in left_parts.iter_mut().enumerate() {
            // TODO: revisit if we need to change the branch order for perf.
            if let Some(code) = self.codes.iter().position(|v| *v == *left) {
                *left = code as u16;
            } else {
                exceptions.push(*left);
                exceptions_pos.push(idx as _);

                *left = 0u16;
            }
        }

        // Bit-pack down the encoded left-parts array that have been dictionary encoded.
        let primitive_left = PrimitiveArray::new(left_parts, array.validity().clone());
        // SAFETY: by construction, all values in left_parts can be packed to left_bit_width.
        let packed_left = unsafe {
            bitpack_encode_unchecked(primitive_left, left_bit_width as _)
                .vortex_unwrap()
                .into_array()
        };

        let primitive_right = PrimitiveArray::new(right_parts, Validity::NonNullable);
        // SAFETY: by construction, all values in right_parts are right_bit_width + leading zeros.
        let packed_right = unsafe {
            bitpack_encode_unchecked(primitive_right, self.right_bit_width as _)
                .vortex_unwrap()
                .into_array()
        };

        // Bit-pack the dict-encoded left-parts
        // Bit-pack the right-parts
        // Patches for exceptions.
        let exceptions = (!exceptions_pos.is_empty()).then(|| {
            let max_exc_pos = exceptions_pos.last().copied().unwrap_or_default();
            let bw = bit_width!(max_exc_pos) as u8;

            let exc_pos_array = PrimitiveArray::new(exceptions_pos, Validity::NonNullable);
            // SAFETY: We calculate bw such that it is wide enough to hold the largest position index.
            let packed_pos = unsafe {
                bitpack_encode_unchecked(exc_pos_array, bw)
                    .vortex_unwrap()
                    .into_array()
            };

            Patches::new(doubles.len(), 0, packed_pos, exceptions.into_array())
        });

        ALPRDArray::try_new(
            DType::Primitive(T::PTYPE, packed_left.dtype().nullability()),
            packed_left,
            Buffer::<u16>::copy_from(&self.codes),
            packed_right,
            self.right_bit_width,
            exceptions,
        )
        .vortex_expect("ALPRDArray construction in encode")
    }
}

/// Decode a vector of ALP-RD encoded values back into their original floating point format.
///
/// # Panics
///
/// The function panics if the provided `left_parts` and `right_parts` differ in length.
pub fn alp_rd_decode<T: ALPRDFloat>(
    left_parts: Buffer<u16>,
    left_parts_dict: &[u16],
    right_bit_width: u8,
    right_parts: BufferMut<T::UINT>,
    left_parts_patches: Option<&Patches>,
) -> VortexResult<Buffer<T>> {
    if left_parts.len() != right_parts.len() {
        vortex_bail!("alp_rd_decode: left_parts.len != right_parts.len");
    }

    // Decode the left-parts dictionary
    let mut values = BufferMut::<u16>::from_iter(
        left_parts
            .iter()
            .map(|code| left_parts_dict[*code as usize]),
    );

    // Apply any patches
    if let Some(patches) = left_parts_patches {
        let indices = patches.indices().to_primitive()?;
        let patch_values = patches.values().to_primitive()?;
        match_each_integer_ptype!(indices.ptype(), |$T| {
            indices
                .as_slice::<$T>()
                .iter()
                .copied()
                .map(|idx| idx - patches.offset() as $T)
                .zip(patch_values.as_slice::<u16>().iter())
                .for_each(|(idx, v)| values[idx as usize] = *v);
        })
    }

    // Shift the left-parts and add in the right-parts.
    let mut index = 0;
    Ok(right_parts
        .map_each(|right| {
            let left = values[index];
            index += 1;
            let left = <T as ALPRDFloat>::from_u16(left);
            T::from_bits((left << (right_bit_width as usize)) | right)
        })
        .freeze())
}

/// Find the best "cut point" for a set of floating point values such that we can
/// cast them all to the relevant value instead.
fn find_best_dictionary<T: ALPRDFloat>(samples: &[T]) -> ALPRDDictionary {
    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();

    for p in 1..=16 {
        let candidate_right_bw = (T::BITS - p) as u8;
        let (dictionary, exception_count) =
            build_left_parts_dictionary::<T>(samples, candidate_right_bw, MAX_DICT_SIZE);
        let estimated_size = estimate_compression_size(
            dictionary.right_bit_width,
            dictionary.left_bit_width,
            exception_count,
            samples.len(),
        );
        if estimated_size < best_est_size {
            best_est_size = estimated_size;
            best_dict = dictionary;
        }
    }

    best_dict
}

/// Build dictionary of the leftmost bits.
fn build_left_parts_dictionary<T: ALPRDFloat>(
    samples: &[T],
    right_bw: u8,
    max_dict_size: u8,
) -> (ALPRDDictionary, usize) {
    assert!(
        right_bw >= (T::BITS - CUT_LIMIT) as _,
        "left-parts must be <= 16 bits"
    );

    // Count the number of occurrences of each left bit pattern
    let counts = samples
        .iter()
        .copied()
        .map(|v| <T as ALPRDFloat>::to_u16(T::to_bits(v).shr(right_bw as _)))
        .counts();

    // Sorted counts: sort by negative count so that heavy hitters sort first.
    let mut sorted_bit_counts: Vec<(u16, usize)> = counts.into_iter().collect_vec();
    sorted_bit_counts.sort_by_key(|(_, count)| count.wrapping_neg());

    // Assign the most-frequently occurring left-bits as dictionary codes, up to `dict_size`...
    let mut dictionary = HashMap::with_capacity_and_hasher(max_dict_size as _, FxBuildHasher);
    let mut code = 0u16;
    while code < (max_dict_size as _) && (code as usize) < sorted_bit_counts.len() {
        let (bits, _) = sorted_bit_counts[code as usize];
        dictionary.insert(bits, code);
        code += 1;
    }

    // ...and the rest are exceptions.
    let exception_count: usize = sorted_bit_counts
        .iter()
        .skip(code as _)
        .map(|(_, count)| *count)
        .sum();

    // Left bit-width is determined based on the actual dictionary size.
    let max_code = dictionary.len() - 1;
    let left_bw = bit_width!(max_code) as u8;

    (
        ALPRDDictionary {
            dictionary,
            right_bit_width: right_bw,
            left_bit_width: left_bw,
        },
        exception_count,
    )
}

/// Estimate the bits-per-value when using these compression settings.
fn estimate_compression_size(
    right_bw: u8,
    left_bw: u8,
    exception_count: usize,
    sample_n: usize,
) -> f64 {
    const EXC_POSITION_SIZE: usize = 16; // two bytes for exception position.
    const EXC_SIZE: usize = 16; // two bytes for each exception (up to 16 front bits).

    let exceptions_size = exception_count * (EXC_POSITION_SIZE + EXC_SIZE);
    (right_bw as f64) + (left_bw as f64) + ((exceptions_size as f64) / (sample_n as f64))
}

/// The ALP-RD dictionary, encoding the "left parts" and their dictionary encoding.
#[derive(Debug, Default)]
struct ALPRDDictionary {
    /// Items in the dictionary are bit patterns, along with their 16-bit encoding.
    dictionary: HashMap<u16, u16, FxBuildHasher>,
    /// The (compressed) left bit width. This is after bit-packing the dictionary codes.
    left_bit_width: u8,
    /// The right bit width. This is the bit-packed width of each of the "real double" values.
    right_bit_width: u8,
}
