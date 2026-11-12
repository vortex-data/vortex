// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::x86_64::__m128i;
use std::arch::x86_64::__m256;
use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm256_add_epi32;
use std::arch::x86_64::_mm256_and_si256;
use std::arch::x86_64::_mm256_castsi256_ps;
use std::arch::x86_64::_mm256_cmpeq_epi32;
use std::arch::x86_64::_mm256_cvtepu16_epi32;
use std::arch::x86_64::_mm256_i32gather_epi32;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_mask_i32gather_epi32;
use std::arch::x86_64::_mm256_max_epu32;
use std::arch::x86_64::_mm256_min_epu32;
use std::arch::x86_64::_mm256_movemask_ps;
use std::arch::x86_64::_mm256_set1_epi32;
use std::arch::x86_64::_mm256_setr_epi32;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_slli_epi32;
use std::arch::x86_64::_mm256_srli_epi32;
use std::arch::x86_64::_mm256_srlv_epi32;
use std::slice::from_raw_parts;

use fearless_simd::Level;
use fearless_simd::Simd;
use num_traits::AsPrimitive;

use crate::BitBuffer;
use crate::BitBufferView;
use crate::BufferMut;
use crate::bit::get_bit;
use crate::bit::get_bit_unchecked;

/// Caller must verify indices.len() == validity.len()
pub(super) fn take<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
    validity: Option<BitBufferView<'_>>,
) -> Option<BitBuffer> {
    let avx2 = Level::new().as_avx2()?;
    if bits.len() >= (1 << 31) - 8 {
        return None;
    }
    let last_dword = (bits.offset() + bits.len() - 1) / 32;
    if (last_dword + 1) * 4 > bits.inner().len() {
        return None;
    }

    let size = size_of::<I>();
    if size == size_of::<u16>() {
        // SAFETY: if no validity is provided, caller has verified the indices
        // are in bounds of "bits". If validity is provided, we clamp every
        // index so it never produces an out of bound reads and panic if any
        // clamped index isn't equal to original index.
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u16>(), indices.len()) };
        let indices_ptr: *const u16 = indices.as_ptr();
        unsafe {
            let group = |group_idx| -> __m256i {
                let group_ptr: *const __m128i = indices_ptr.add(group_idx * 8).cast();
                // copy 128 bits into __m128i
                let vector: __m128i = _mm_loadu_si128(group_ptr);
                // zero-extend unsigned 16-bit integers in __m128i to signed
                // 32-bit integers in __m256i
                _mm256_cvtepu16_epi32(vector)
            };
            return Some(avx2.vectorize(|| take_group(bits, indices, validity, group)));
        }
    }
    if size == size_of::<u32>() {
        // SAFETY: see u16 case above
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u32>(), indices.len()) };
        let indices_ptr: *const u32 = indices.as_ptr();
        unsafe {
            let group = |group_idx| -> __m256i {
                let group_ptr: *const __m256i = indices_ptr.add(group_idx * 8).cast();
                // copy 256 bits into __m256i
                _mm256_loadu_si256(group_ptr)
            };
            return Some(avx2.vectorize(|| take_group(bits, indices, validity, group)));
        }
    }
    None
}

/// By the time we call this function we already handle cases of u16 and u32
/// separately as there's different packing code. Now we also need to branch
/// on validity. take(), the first function, is called from two places, one
/// with validity and one without. Consequently, there are different gather
/// instructions depending on whether validity is present.
#[target_feature(enable = "avx2")]
unsafe fn take_group<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
    validity: Option<BitBufferView<'_>>,
    load_group: impl Fn(usize) -> __m256i,
) -> BitBuffer {
    let inner = bits.inner();
    let base = inner.as_ptr();
    let zero: __m256i = _mm256_setzero_si256();

    let Some(validity) = validity else {
        // If validity isn't present, caller has verified "indices" are valid
        // offsets into "bits" (see min/max code in take.rs), so we can use
        // unchecked access.
        let gather = |_group_idx: usize, indices: __m256i| -> (__m256i, __m256i) {
            let gathered = unsafe { _mm256_i32gather_epi32::<4>(base.cast(), indices) };
            (gathered, zero)
        };
        // inside take_lanes we do min(x, max_index) to check out of bound
        // reads. Without validity there can't be out of bound reads so we
        // pass u32::MAX. LLVM therefore reduces min(x, MAX) to x.
        return unsafe { take_lanes(bits, indices, load_group, gather, usize::MAX, None) };
    };

    // If validity is present, we can't trust "indices" are valid offsets.
    // There also may be garbage under a NULL index, and we need to avoid
    // reading bits's offset at this garbage.
    let valid_bytes = validity.inner();
    let validity_offset = validity.offset();
    let validity_shift = validity_offset % 8;
    let validity_first_byte = validity_offset / 8;
    let has_non_byte_shift: usize = (validity_shift != 0).as_();

    let lane_bits: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    let gather = |group_idx: usize, indices: __m256i| -> (__m256i, __m256i) {
        let byte = validity_first_byte + group_idx;
        let next_byte = byte + has_non_byte_shift;

        // We've verified in take.rs validity read is in bounds
        let lo = unsafe { *valid_bytes.get_unchecked(byte) as u16 };
        let hi = unsafe { *valid_bytes.get_unchecked(next_byte) as u16 };

        let window = lo | (hi << 8);
        let valid = i32::from(window >> validity_shift);

        // mask = "valid" in every element of vector
        let mask = _mm256_set1_epi32(valid);
        // selected = mask & lane_bits
        let selected = _mm256_and_si256(mask, lane_bits);
        // lanes = selected == lane_bits
        let lanes = _mm256_cmpeq_epi32(selected, lane_bits);

        let gathered =
            unsafe { _mm256_mask_i32gather_epi32::<4>(zero, base.cast(), indices, lanes) };
        (gathered, lanes)
    };

    unsafe {
        take_lanes(
            bits,
            indices,
            load_group,
            gather,
            bits.len() - 1,
            Some((valid_bytes, validity_offset)),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn take_lanes<I>(
    bits: BitBufferView<'_>,
    indices: &[I],
    load_group: impl Fn(usize) -> __m256i,
    gather: impl Fn(usize, __m256i) -> (__m256i, __m256i),
    max_index: usize,
    validity: Option<(&[u8], usize)>,
) -> BitBuffer
where
    I: AsPrimitive<usize>,
{
    let total = indices.len();
    let groups = total / 8;

    let bit_offset: __m256i = _mm256_set1_epi32(bits.offset().as_());
    let low_bits: __m256i = _mm256_set1_epi32(31);
    let max_index: __m256i = _mm256_set1_epi32(max_index.as_());
    let mut max_taken_index: __m256i = _mm256_setzero_si256();

    let mut out = BufferMut::<u8>::with_capacity(total.div_ceil(8));
    for group in 0..groups {
        let original_indices: __m256i = load_group(group);
        // clamped = min(original_indices, max_index)
        // "clamped" never reads beyond bits's buffer end
        let clamped = _mm256_min_epu32(original_indices, max_index);
        // clamped_shifted = clamped + bit_offset
        let clamped_shifted = _mm256_add_epi32(clamped, bit_offset);

        // gather_indices = clamped_shifted / 32
        // Gather can read only 32-bit words
        let gather_indices = _mm256_srli_epi32::<5>(clamped_shifted);
        let (gathered, lanes) = gather(group, gather_indices);
        // taken_indices = original_indices & lanes
        let taken_indices = _mm256_and_si256(original_indices, lanes);
        // max_taken_index = max(max_taken_index, taken_indices)
        max_taken_index = _mm256_max_epu32(max_taken_index, taken_indices);

        // shift = clamped_shifted % 32
        // Bit we need is now in position 0
        let shift = _mm256_and_si256(clamped_shifted, low_bits);
        // shifted = gathered >> shift
        let shifted = _mm256_srlv_epi32(gathered, shift);
        // top = shifted << 31
        // Bit we need is now in position 31
        let top: __m256i = _mm256_slli_epi32::<31>(shifted);
        // AVX2 doesn't have integral MSB for 32bit lanes, cast to floats
        let top: __m256 = _mm256_castsi256_ps(top);
        // per-element 1 if MSB is set, 0 otherwise
        let msb: u8 = _mm256_movemask_ps(top).as_();

        // SAFETY: out has sufficient capacity
        unsafe { out.push_unchecked(msb) };
    }

    // max_clamped = min(max_taken_index, max_index)
    let max_clamped = _mm256_min_epu32(max_taken_index, max_index);
    // in_bounds = max_clamped == max_taken_index
    let in_bounds: __m256i = _mm256_cmpeq_epi32(max_clamped, max_taken_index);
    // AVX2 doesn't have integral MSB for 32bit lanes, cast to floats
    let in_bounds: __m256 = _mm256_castsi256_ps(in_bounds);
    // per-element 1 if MSB is set, 0 otherwise
    let msb = _mm256_movemask_ps(in_bounds);
    assert_eq!(msb, 0xFF, "index out of bounds");

    let tail = total % 8;
    if tail == 0 {
        return BitBuffer::new(out.freeze(), total);
    }

    let start = groups * 8;
    let inner = bits.inner();
    let mut byte = 0u8;
    for bit in 0..tail {
        let pos = start + bit;
        let keep = validity.map_or(usize::MAX, |(valid_bytes, validity_offset)| {
            // SAFETY: validity has same length as indices
            (unsafe { get_bit_unchecked(valid_bytes.as_ptr(), validity_offset + pos) } as usize)
                .wrapping_neg()
        });
        // SAFETY: pos stays within indices
        let idx = unsafe { indices.get_unchecked(pos) }.as_() & keep;
        let value = get_bit(inner, bits.offset() + idx);
        byte |= (value as u8) << bit;
    }
    // SAFETY: out has sufficient capacity
    unsafe { out.push_unchecked(byte) }
    BitBuffer::new(out.freeze(), total)
}
