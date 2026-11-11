// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP-RD decoding fused with the unpacking of its bit-packed parts.
//!
//! The general decode unpacks the left and right parts into two full-length buffers, then makes
//! further passes to look the left codes up and combine them with the right parts. For a large
//! chunk those buffers fall out of cache between passes. When both parts are plainly bit-packed,
//! this decodes them together one 1024-value block at a time, so each value is unpacked, looked up
//! and combined while it is still in cache, and only the output is written to memory.

use ::alp::ALPRDFloat;
use ::alp::MAX_DICT_SIZE;
use fastlanes::BitPacking;
use num_traits::Bounded;
use num_traits::Zero;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitWidthsView;
use vortex_fastlanes::FL_CHUNK_SIZE;

use super::ALPRD;
use super::ALPRDArrayExt;
use super::array::ALPRDArraySlotsExt;
use super::code_not_in_dictionary;

/// Decodes `array` block by block if both its parts are bit-packed at one width with no patches
/// of their own and no nulls, or returns `None` for the general decode to handle it.
pub(super) fn fused_decode(
    array: &Array<ALPRD>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(left), Some(right)) = (
        array.left_parts().as_opt::<BitPacked>(),
        array.right_parts().as_opt::<BitPacked>(),
    ) else {
        return Ok(None);
    };
    let validity = left.validity()?;
    if !matches!(validity, Validity::NonNullable | Validity::AllValid)
        || !matches!(
            right.validity()?,
            Validity::NonNullable | Validity::AllValid
        )
        || left.patches().is_some()
        || right.patches().is_some()
        || left.offset() != right.offset()
        || array.left_parts_dictionary().len() > usize::from(MAX_DICT_SIZE)
    {
        return Ok(None);
    }
    let (BitWidthsView::Global(left_width), BitWidthsView::Global(right_width)) =
        (left.bit_widths(), right.bit_widths())
    else {
        return Ok(None);
    };
    let parts = Parts {
        left: left.packed_slice::<u16>(),
        left_width: usize::from(left_width),
        offset: usize::from(left.offset()),
        len: array.len(),
        dict: array.left_parts_dictionary().as_slice(),
        shift: array.right_bit_width(),
    };
    let patches = array.left_parts_patches();
    let decoded = match array.dtype().as_ptype() {
        PType::F32 => decode::<f32>(
            &parts,
            right.packed_slice::<u32>(),
            usize::from(right_width),
            patches,
            ctx,
        )?
        .map(|values| PrimitiveArray::new(values.freeze(), validity).into_array()),
        PType::F64 => decode::<f64>(
            &parts,
            right.packed_slice::<u64>(),
            usize::from(right_width),
            patches,
            ctx,
        )?
        .map(|values| PrimitiveArray::new(values.freeze(), validity).into_array()),
        _ => None,
    };
    Ok(decoded)
}

/// What decoding needs besides the right parts, whose type depends on the float width.
struct Parts<'a> {
    left: &'a [u16],
    left_width: usize,
    offset: usize,
    len: usize,
    dict: &'a [u16],
    shift: u8,
}

fn decode<T>(
    parts: &Parts<'_>,
    right: &[T::UINT],
    right_width: usize,
    patches: Option<Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<BufferMut<T>>>
where
    T: ALPRDFloat + NativePType,
    T::UINT: BitPacking + NativePType + Combine,
{
    let total = parts.offset + parts.len;
    let blocks = total.div_ceil(FL_CHUNK_SIZE);
    let left_per_block = FL_CHUNK_SIZE * parts.left_width / u16::BITS as usize;
    let right_per_block = FL_CHUNK_SIZE * right_width / (8 * size_of::<T::UINT>());
    if parts.left.len() < blocks * left_per_block || right.len() < blocks * right_per_block {
        return Ok(None);
    }

    let shift = usize::from(parts.shift);
    let mut shifted = [T::UINT::zero(); DICT];
    for (slot, &entry) in shifted.iter_mut().zip(parts.dict) {
        *slot = <T as ALPRDFloat>::from_u16(entry) << shift;
    }
    let combine = T::UINT::combiner();

    let mut out = BufferMut::<T::UINT>::with_capacity(parts.len);
    let mut codes = [0u16; FL_CHUNK_SIZE];
    let mut rights = [T::UINT::zero(); FL_CHUNK_SIZE];
    let mut max_code = 0u16;
    for block in 0..blocks {
        // SAFETY: each packed block holds `FL_CHUNK_SIZE * width / bits` values, checked to be
        // in bounds above, and each output holds `FL_CHUNK_SIZE` values.
        unsafe {
            BitPacking::unchecked_unpack(
                parts.left_width,
                &parts.left[block * left_per_block..][..left_per_block],
                &mut codes,
            );
            BitPacking::unchecked_unpack(
                right_width,
                &right[block * right_per_block..][..right_per_block],
                &mut rights,
            );
        }
        let start = if block == 0 { parts.offset } else { 0 };
        let end = (total - block * FL_CHUNK_SIZE).min(FL_CHUNK_SIZE);
        max_code = codes[start..end]
            .iter()
            .fold(max_code, |max, &code| max.max(code));
        combine(&codes, &mut rights, &shifted);
        out.extend_from_slice(&rights[start..end]);
    }
    if usize::from(max_code) >= parts.dict.len() {
        return Err(code_not_in_dictionary(max_code, parts.dict.len()));
    }

    if let Some(patches) = patches {
        let right_mask = !(T::UINT::max_value() << shift);
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let values = values.as_slice::<u16>();
        let offset = patches.offset();
        let out = out.as_mut_slice();
        match_each_integer_ptype!(indices.ptype(), |I| {
            for (&index, &value) in indices.as_slice::<I>().iter().zip(values) {
                #[allow(clippy::unnecessary_cast)]
                let at = index as usize - offset;
                out[at] = (<T as ALPRDFloat>::from_u16(value) << shift) | (out[at] & right_mask);
            }
        });
    }

    // SAFETY: every bit pattern of `T::UINT` is a valid `T` (u32 to f32, u64 to f64).
    Ok(Some(unsafe { out.transmute::<T>() }))
}

/// A block of codes and right parts, combined in place into the bits of the original floats.
type CombineFn<U> = fn(&[u16; FL_CHUNK_SIZE], &mut [U; FL_CHUNK_SIZE], &[U; DICT]);

const DICT: usize = MAX_DICT_SIZE as usize;

/// Codes are checked against the dictionary once the chunk is decoded; until then this mask keeps
/// every lookup in bounds.
const CODE_MASK: usize = DICT - 1;

/// The unsigned type of a float's bits, which picks the fastest way to combine a block.
trait Combine: Sized + Copy + std::ops::BitOr<Output = Self> {
    fn combiner() -> CombineFn<Self> {
        combine_scalar::<Self>
    }
}

impl Combine for u64 {}

impl Combine for u32 {
    fn combiner() -> CombineFn<Self> {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return combine_u32_avx2_checked;
        }
        combine_scalar::<Self>
    }
}

fn combine_scalar<U: Copy + std::ops::BitOr<Output = U>>(
    codes: &[u16; FL_CHUNK_SIZE],
    rights: &mut [U; FL_CHUNK_SIZE],
    shifted: &[U; DICT],
) {
    for (right, &code) in rights.iter_mut().zip(codes) {
        *right = shifted[usize::from(code) & CODE_MASK] | *right;
    }
}

/// Only handed out by [`Combine::combiner`] once the CPU is known to support AVX2.
#[cfg(target_arch = "x86_64")]
fn combine_u32_avx2_checked(
    codes: &[u16; FL_CHUNK_SIZE],
    rights: &mut [u32; FL_CHUNK_SIZE],
    shifted: &[u32; DICT],
) {
    // SAFETY: this is only returned by `Combine::combiner` after detecting AVX2.
    unsafe { combine_u32_avx2(codes, rights, shifted) }
}

/// Eight lookups at a time: the dictionary fits one register, and a lane permute indexes it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn combine_u32_avx2(
    codes: &[u16; FL_CHUNK_SIZE],
    rights: &mut [u32; FL_CHUNK_SIZE],
    shifted: &[u32; DICT],
) {
    use std::arch::x86_64::*;

    // SAFETY: `shifted` holds eight `u32`s, one unaligned 256-bit load.
    let table = unsafe { _mm256_loadu_si256(shifted.as_ptr().cast()) };
    let mask = _mm256_set1_epi32(CODE_MASK as i32);
    for (codes, rights) in codes
        .as_chunks::<8>()
        .0
        .iter()
        .zip(rights.as_chunks_mut::<8>().0)
    {
        // SAFETY: each chunk is eight `u16`s (128 bits) or eight `u32`s (256 bits), loaded and
        // stored unaligned.
        unsafe {
            let codes = _mm256_cvtepu16_epi32(_mm_loadu_si128(codes.as_ptr().cast()));
            let lefts = _mm256_permutevar8x32_epi32(table, _mm256_and_si256(codes, mask));
            let values = _mm256_loadu_si256(rights.as_ptr().cast());
            _mm256_storeu_si256(rights.as_mut_ptr().cast(), _mm256_or_si256(lefts, values));
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ALPRD;
    use crate::RDEncoder;
    use crate::RDEncoderExt;

    /// Floats with random low bits, so most need ALP-RD, and a few unusual exponents that become
    /// left-part patches.
    fn values(count: usize) -> Vec<f64> {
        let mut state = 7u64;
        (0..count)
            .map(|idx| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = f64::from(u32::try_from(state >> 32).unwrap_or(0)) / f64::from(u32::MAX);
                if idx % 97 == 0 {
                    unit * 1e30
                } else {
                    unit - 0.5
                }
            })
            .collect()
    }

    #[rstest]
    #[case::whole(0..5000)]
    #[case::sliced_in_a_block(100..900)]
    #[case::sliced_across_blocks(1000..4321)]
    fn fused_decode_round_trips_f32(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
        #[expect(clippy::cast_possible_truncation, reason = "test floats")]
        let floats: Vec<f32> = values(5000).into_iter().map(|v| v as f32).collect();
        let array = PrimitiveArray::from_iter(floats.iter().copied());
        let encoded = RDEncoder::new(&floats).encode(array.as_view()).into_array();
        assert!(encoded.as_opt::<ALPRD>().is_some());
        let session = VortexSession::empty();
        crate::initialize(&session);
        let sliced = encoded.slice(range.clone())?;
        let expected = PrimitiveArray::from_iter(floats[range].iter().copied());
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::whole(0..5000)]
    #[case::sliced_across_blocks(1000..4321)]
    fn fused_decode_round_trips_f64(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
        let floats = values(5000);
        let array = PrimitiveArray::from_iter(floats.iter().copied());
        let encoded = RDEncoder::new(&floats).encode(array.as_view()).into_array();
        let session = VortexSession::empty();
        crate::initialize(&session);
        let sliced = encoded.slice(range.clone())?;
        let expected = PrimitiveArray::from_iter(floats[range].iter().copied());
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }
}
