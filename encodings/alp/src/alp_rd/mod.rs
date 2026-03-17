// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

pub use array::*;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;

mod array;
mod compute;
mod kernel;
mod ops;
mod rules;
mod slice;

use std::ops::Shl;
use std::ops::Shr;

use num_traits::Float;
use num_traits::One;
use num_traits::PrimInt;
use num_traits::Zero;
use rustc_hash::FxBuildHasher;
use vortex_array::DynArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

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

const MAX_DICT_SIZE: u8 = 16;

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
/// up to [`MAX_DICT_SIZE`] elements. Each double can then be broken into the front/left `L` bits, which neatly
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
    /// Flat lookup table: index by left_bits u16 value, value is (code + 1) or 0 for not-in-dict.
    /// Using a sentinel of 0 avoids Option overhead and branch mispredictions in the hot loop.
    /// Max left_bits is 16 bits = 65536 entries, at 2 bytes each = 128KB.
    reverse_flat: Vec<u16>,
}

/// Sentinel value indicating a left_bits value is not in the dictionary.
const NOT_IN_DICT: u16 = 0;

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

        Self::from_parts(dictionary.right_bit_width, codes)
    }

    /// Build a new encoder from known parameters.
    pub fn from_parts(right_bit_width: u8, codes: Vec<u16>) -> Self {
        // Build a flat lookup table for O(1) reverse lookup during encoding.
        // Left parts are at most CUT_LIMIT (16) bits wide, so the table is at most 64K entries.
        let table_size = 1usize << CUT_LIMIT;
        let mut reverse_flat = vec![NOT_IN_DICT; table_size];
        for (code, &bits) in codes.iter().enumerate() {
            if (bits as usize) < table_size {
                // Store code + 1 so that 0 remains the sentinel for "not in dict".
                reverse_flat[bits as usize] = (code as u16) + 1;
            }
        }

        Self {
            right_bit_width,
            codes,
            reverse_flat,
        }
    }

    /// Encode a set of floating point values with ALP-RD.
    ///
    /// Each value will be split into a left and right component, which are compressed individually.
    // TODO(joe): make fallible
    pub fn encode(&self, array: &PrimitiveArray) -> ALPRDArray {
        match_each_alp_float_ptype!(array.ptype(), |P| { self.encode_generic::<P>(array) })
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
        let n = doubles.len();

        let mut right_parts: BufferMut<T::UINT> = BufferMut::with_capacity(n);
        let mut exceptions_pos: BufferMut<u64> = BufferMut::with_capacity(n / 4);
        let mut exceptions: BufferMut<u16> = BufferMut::with_capacity(n / 4);

        let right_mask = T::UINT::one().shl(self.right_bit_width as _) - T::UINT::one();
        let max_code = self.codes.len() - 1;
        let left_bit_width = bit_width!(max_code);
        let rbw = self.right_bit_width as usize;

        // Vectorizable pass: extract right parts via bit masking.
        right_parts.extend_trusted(doubles.iter().map(|&v| T::to_bits(v) & right_mask));

        // Pre-allocate left_parts at full size so we can write via direct indexing.
        let mut left_parts: BufferMut<u16> = BufferMut::zeroed(n);
        let left_slice = left_parts.as_mut_slice();

        // Dict-encode left parts and collect exceptions.
        let reverse_flat = &self.reverse_flat;
        for (idx, &v) in doubles.iter().enumerate() {
            let left = <T as ALPRDFloat>::to_u16(T::to_bits(v).shr(rbw));
            // SAFETY: left is at most 16 bits, reverse_flat has 2^16 entries.
            let lookup = unsafe { *reverse_flat.get_unchecked(left as usize) };
            if lookup != NOT_IN_DICT {
                // SAFETY: idx < n and left_slice has exactly n elements.
                unsafe { *left_slice.get_unchecked_mut(idx) = lookup - 1 };
            } else {
                // left_slice[idx] is already 0 from zeroed allocation.
                exceptions.push(left);
                exceptions_pos.push(idx as _);
            }
        }

        // Bit-pack down the encoded left-parts array that have been dictionary encoded.
        let primitive_left = PrimitiveArray::new(left_parts, array.validity().clone());
        // SAFETY: by construction, all values in left_parts can be packed to left_bit_width.
        let packed_left = unsafe {
            bitpack_encode_unchecked(primitive_left, left_bit_width as _)
                .vortex_expect("bitpack_encode_unchecked should succeed for left parts")
                .into_array()
        };

        let primitive_right = PrimitiveArray::new(right_parts, Validity::NonNullable);
        // SAFETY: by construction, all values in right_parts are right_bit_width + leading zeros.
        let packed_right = unsafe {
            bitpack_encode_unchecked(primitive_right, self.right_bit_width as _)
                .vortex_expect("bitpack_encode_unchecked should succeed for right parts")
                .into_array()
        };

        let exceptions = (!exceptions_pos.is_empty()).then(|| {
            let max_exc_pos = exceptions_pos.last().copied().unwrap_or_default();
            let bw = bit_width!(max_exc_pos) as u8;

            let exc_pos_array = PrimitiveArray::new(exceptions_pos, Validity::NonNullable);
            // SAFETY: We calculate bw such that it is wide enough to hold the largest position index.
            let packed_pos = unsafe {
                bitpack_encode_unchecked(exc_pos_array, bw)
                    .vortex_expect(
                        "bitpack_encode_unchecked should succeed for exception positions",
                    )
                    .into_array()
            };

            Patches::new(
                doubles.len(),
                0,
                packed_pos,
                exceptions.into_array(),
                // TODO(0ax1): handle chunk offsets
                None,
            )
            .vortex_expect("Patches construction in encode")
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
    mut right_parts: BufferMut<T::UINT>,
    left_parts_patches: Option<&Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    if left_parts.len() != right_parts.len() {
        vortex_panic!("alp_rd_decode: left_parts.len != right_parts.len");
    }

    let shift = right_bit_width as usize;

    // Build a pre-shifted dictionary on the stack (max 16 entries) to avoid heap allocation.
    let dict_len = left_parts_dict.len();
    let mut dict_shifted = [T::UINT::zero(); MAX_DICT_SIZE as usize];
    for (i, &v) in left_parts_dict.iter().enumerate() {
        dict_shifted[i] = <T as ALPRDFloat>::from_u16(v) << shift;
    }

    // Dict-lookup the left code, OR into right_parts in place.
    let right_slice = right_parts.as_mut_slice();
    let dict_ptr = dict_shifted.as_ptr();
    right_slice
        .iter_mut()
        .zip(left_parts.iter())
        .for_each(|(right, &code)| {
            // SAFETY: codes are dictionary-encoded indices guaranteed to be < dict_len.
            debug_assert!((code as usize) < dict_len);
            let shifted = unsafe { *dict_ptr.add(code as usize) };
            *right = shifted | *right;
        });

    // Apply any patches (patch values are raw u16, so we widen and shift them).
    if let Some(patches) = left_parts_patches {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        alp_rd_apply_patches::<T>(
            right_parts.as_mut_slice(),
            &indices,
            &patch_values,
            patches.offset(),
            shift,
        );
    }

    // Reinterpret bits as floats via zero-cost transmute instead of per-element from_bits.
    // SAFETY: T::UINT and T have the same size and alignment (u32/f32, u64/f64),
    // and all bit patterns were originally encoded from valid floats.
    Ok(unsafe { right_parts.transmute::<T>() }.freeze())
}

/// Apply patches directly to the combined (left|right) buffer.
///
/// Patch values are raw u16 left-parts that need to be widened and shifted,
/// then OR'd with the existing right-part bits already in the buffer.
fn alp_rd_apply_patches<F: ALPRDFloat>(
    combined: &mut [F::UINT],
    indices: &PrimitiveArray,
    patch_values: &PrimitiveArray,
    offset: usize,
    shift: usize,
) {
    // The right_mask extracts the right-part bits that are already in the combined buffer.
    let right_mask = F::UINT::one().shl(shift) - F::UINT::one();

    match_each_integer_ptype!(indices.ptype(), |T| {
        indices
            .as_slice::<T>()
            .iter()
            .copied()
            .map(|idx| idx - offset as T)
            .zip(patch_values.as_slice::<u16>().iter())
            .for_each(|(idx, v)| {
                let i = idx as usize;
                // Overwrite the left-part bits while preserving the right-part bits.
                combined[i] =
                    (<F as ALPRDFloat>::from_u16(*v) << shift) | (combined[i] & right_mask);
            });
    })
}

/// Threshold below which we use the HashMap-based path (avoids the cost of
/// zeroing a 256KB flat array for very small samples).
const FLAT_ARRAY_THRESHOLD: usize = 2048;

/// Find the best "cut point" for a set of floating point values such that we can
/// cast them all to the relevant value instead.
///
/// Extracts the top 16 bits (left parts at p=16) from each sample value once, then
/// iterates from p=16 down to p=1. For large samples, frequency counts are stored in
/// a flat array indexed by bit pattern and folded between iterations (single scan).
/// For small samples, a HashMap is used per iteration to avoid the 256KB allocation.
fn find_best_dictionary<T: ALPRDFloat>(samples: &[T]) -> ALPRDDictionary {
    let max_p = CUT_LIMIT;
    let min_right_bw = T::BITS - max_p;

    // Extract the top CUT_LIMIT bits from each sample value once. All subsequent
    // iterations derive their left parts by right-shifting these u16 values.
    let left_parts: Vec<u16> = samples
        .iter()
        .map(|&v| <T as ALPRDFloat>::to_u16(T::to_bits(v).shr(min_right_bw as _)))
        .collect();

    if samples.len() >= FLAT_ARRAY_THRESHOLD {
        find_best_dictionary_flat(&left_parts, samples.len(), T::BITS)
    } else {
        find_best_dictionary_small(&left_parts, samples.len(), T::BITS)
    }
}

/// Large-sample path: uses a flat `[u32; 65536]` array for O(1) frequency counting.
/// Counts for smaller p values are derived by folding adjacent pairs, so the sample
/// is only scanned once.
fn find_best_dictionary_flat(
    left_parts: &[u16],
    sample_len: usize,
    type_bits: usize,
) -> ALPRDDictionary {
    let max_p = CUT_LIMIT;
    let max_domain = 1usize << max_p;

    let mut counts = vec![0u32; max_domain];
    for &left in left_parts {
        counts[left as usize] += 1;
    }

    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();
    let mut bit_counts_buf: Vec<(u16, u32)> = Vec::with_capacity(sample_len.min(max_domain));

    for p in (1..=max_p).rev() {
        let domain = 1usize << p;
        let right_bw = (type_bits - p) as u8;

        let (dictionary, exception_count) = build_left_parts_dictionary_from_counts(
            &counts[..domain],
            right_bw,
            MAX_DICT_SIZE,
            &mut bit_counts_buf,
        );
        let estimated_size = estimate_compression_size(
            dictionary.right_bit_width,
            dictionary.left_bit_width,
            exception_count,
            sample_len,
        );
        if estimated_size < best_est_size {
            best_est_size = estimated_size;
            best_dict = dictionary;
        }

        if p > 1 {
            let half = domain / 2;
            for i in 0..half {
                counts[i] = counts[2 * i] + counts[2 * i + 1];
            }
        }
    }

    best_dict
}

/// Small-sample path: uses a HashMap for frequency counting to avoid the cost of
/// allocating and zeroing a 256KB flat array.
fn find_best_dictionary_small(
    left_parts: &[u16],
    sample_len: usize,
    type_bits: usize,
) -> ALPRDDictionary {
    let max_p = CUT_LIMIT;
    let dict_size = MAX_DICT_SIZE as usize;

    let mut best_est_size = f64::MAX;
    let mut best_dict = ALPRDDictionary::default();
    let mut counts: HashMap<u16, u32, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(sample_len, FxBuildHasher);
    let mut bit_counts: Vec<(u16, u32)> = Vec::with_capacity(sample_len);

    for p in 1..=max_p {
        let right_bw = (type_bits - p) as u8;
        let shift = (max_p - p) as u32;

        counts.clear();
        for &left in left_parts {
            *counts.entry(left >> shift).or_default() += 1;
        }

        bit_counts.clear();
        bit_counts.extend(counts.iter().map(|(&bits, &c)| (bits, c)));

        if bit_counts.len() > dict_size {
            bit_counts.select_nth_unstable_by_key(dict_size.saturating_sub(1), |(_, count)| {
                count.wrapping_neg()
            });
        }

        let mut dictionary = HashMap::with_capacity_and_hasher(MAX_DICT_SIZE as _, FxBuildHasher);
        let mut exception_count = 0usize;
        for (i, &(bits, count)) in bit_counts.iter().enumerate() {
            if i < dict_size {
                dictionary.insert(bits, i as u16);
            } else {
                exception_count += count as usize;
            }
        }

        let max_code = dictionary.len().saturating_sub(1);
        let left_bw = bit_width!(max_code) as u8;

        let estimated_size =
            estimate_compression_size(right_bw, left_bw, exception_count, sample_len);
        if estimated_size < best_est_size {
            best_est_size = estimated_size;
            best_dict = ALPRDDictionary {
                dictionary,
                right_bit_width: right_bw,
                left_bit_width: left_bw,
            };
        }
    }

    best_dict
}

/// Build a dictionary from pre-computed frequency counts in a flat array.
///
/// `counts` is indexed by left-part bit pattern; `counts[i]` is the frequency of pattern `i`.
/// `bit_counts_buf` is a reusable scratch buffer to avoid per-call allocation.
fn build_left_parts_dictionary_from_counts(
    counts: &[u32],
    right_bw: u8,
    max_dict_size: u8,
    bit_counts_buf: &mut Vec<(u16, u32)>,
) -> (ALPRDDictionary, usize) {
    let dict_size = max_dict_size as usize;

    // Collect only non-zero entries into the reusable buffer.
    bit_counts_buf.clear();
    bit_counts_buf.extend(
        counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(bits, &c)| (bits as u16, c)),
    );

    // Partial sort: only need the top `dict_size` elements by frequency.
    if bit_counts_buf.len() > dict_size {
        bit_counts_buf.select_nth_unstable_by_key(dict_size.saturating_sub(1), |(_, count)| {
            count.wrapping_neg()
        });
    }

    // Build dictionary and count exceptions.
    let mut dictionary = HashMap::with_capacity_and_hasher(max_dict_size as _, FxBuildHasher);
    let mut exception_count = 0usize;
    for (i, &(bits, count)) in bit_counts_buf.iter().enumerate() {
        if i < dict_size {
            dictionary.insert(bits, i as u16);
        } else {
            exception_count += count as usize;
        }
    }

    let max_code = dictionary.len().saturating_sub(1);
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
