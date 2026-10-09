// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::x86_64::__m128i;
use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm256_add_epi32;
use std::arch::x86_64::_mm256_and_si256;
use std::arch::x86_64::_mm256_andnot_si256;
use std::arch::x86_64::_mm256_castsi256_ps;
use std::arch::x86_64::_mm256_cmpeq_epi32;
use std::arch::x86_64::_mm256_cvtepu16_epi32;
use std::arch::x86_64::_mm256_i32gather_epi32;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_mask_i32gather_epi32;
use std::arch::x86_64::_mm256_min_epu32;
use std::arch::x86_64::_mm256_movemask_ps;
use std::arch::x86_64::_mm256_or_si256;
use std::arch::x86_64::_mm256_set1_epi32;
use std::arch::x86_64::_mm256_setr_epi32;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_slli_epi32;
use std::arch::x86_64::_mm256_srli_epi32;
use std::arch::x86_64::_mm256_srlv_epi32;
use std::arch::x86_64::_mm256_testz_si256;
use std::slice::from_raw_parts;

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
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    if bits.len() >= (1 << 31) - 8 {
        return None;
    }
    let last_dword = (bits.offset() + bits.len() - 1) / 32;
    if (last_dword + 1) * 4 > bits.inner().len() {
        return None;
    }

    let size = size_of::<I>();
    if size == size_of::<u16>() {
        // SAFETY: we clamp the index to be valid in take_lanes and check out of
        // bound reads there as well, so any invalid index doesn't cause an out
        // of bounds read and is reported via panic().
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
            return Some(take_group(bits, indices, validity, group));
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
            return Some(take_group(bits, indices, validity, group));
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
        // offsets info "bits" (see min/max code in take.rs), so we can use
        // unchecked access into "bits".
        let gather = |_group: usize, dword: __m256i, _in_bounds: __m256i| -> (__m256i, __m256i) {
            let gathered = unsafe { _mm256_i32gather_epi32::<4>(base.cast(), dword) };
            (gathered, zero)
        };
        return unsafe { take_lanes(bits, indices, load_group, gather, None) };
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

    let gather = |group: usize, dword: __m256i, in_range: __m256i| -> (__m256i, __m256i) {
        let byte = validity_first_byte + group;
        let next_byte = byte + has_non_byte_shift;

        // We've verified in take.rs validity read is in bounds
        let lo = unsafe { *valid_bytes.get_unchecked(byte) as u16 };
        let hi = unsafe { *valid_bytes.get_unchecked(next_byte) as u16 };

        let window = lo | (hi << 8);
        let valid = ((window >> validity_shift) & 0xFF) as u8;

        // all 1 if element is valid, all 0 otherwise
        let valid_vector = _mm256_set1_epi32(i32::from(valid));
        // valid_vector & lane_bits
        let selected = _mm256_and_si256(valid_vector, lane_bits);
        // selected[i] == lane_bits[i]
        let lanes = _mm256_cmpeq_epi32(selected, lane_bits);

        // !in_range & lanes. We have a valid index which is out of bounds
        let violation = _mm256_andnot_si256(in_range, lanes);

        let gathered = unsafe { _mm256_mask_i32gather_epi32::<4>(zero, base.cast(), dword, lanes) };
        (gathered, violation)
    };
    unsafe {
        take_lanes(
            bits,
            indices,
            load_group,
            gather,
            Some((valid_bytes, validity_offset)),
        )
    }
}

#[expect(clippy::cast_possible_truncation)]
#[target_feature(enable = "avx2")]
unsafe fn take_lanes<I>(
    bits: BitBufferView<'_>,
    indices: &[I],
    load_group: impl Fn(usize) -> __m256i,
    gather: impl Fn(usize, __m256i, __m256i) -> (__m256i, __m256i),
    validity: Option<(&[u8], usize)>,
) -> BitBuffer
where
    I: AsPrimitive<usize>,
{
    let total = indices.len();
    let full_groups = total / 8;

    let bit_offset: __m256i = _mm256_set1_epi32(bits.offset() as i32);
    let low_bits: __m256i = _mm256_set1_epi32(31);
    let max_index: __m256i = _mm256_set1_epi32((bits.len() - 1) as i32);
    let mut out_of_bounds: __m256i = _mm256_setzero_si256();

    let mut out = BufferMut::<u8>::with_capacity(total.div_ceil(8));
    for group in 0..full_groups {
        let group_vector: __m256i = load_group(group);

        // group_vector[i] = min(group_vector[i], max_index[i])
        // We clamp every index so it can never read over bits's buffer, and
        // calculate violations separately. If we found a violation after
        // looping< we panic.
        let clamped = _mm256_min_epu32(group_vector, max_index);

        // clamped == group_vector
        let in_range = _mm256_cmpeq_epi32(clamped, group_vector);

        let bitpos = _mm256_add_epi32(clamped, bit_offset);
        let dword = _mm256_srli_epi32::<5>(bitpos);

        let (gathered, violation) = gather(group, dword, in_range);
        out_of_bounds = _mm256_or_si256(out_of_bounds, violation);

        let shift = _mm256_and_si256(bitpos, low_bits);
        let shifted = _mm256_srlv_epi32(gathered, shift);
        let top = _mm256_slli_epi32::<31>(shifted);
        let as_ps = _mm256_castsi256_ps(top);

        let group_bits = _mm256_movemask_ps(as_ps);
        let group_bits: u8 = (group_bits & 0xFF).as_();

        // SAFETY: out has sufficient capacity
        unsafe { out.push_unchecked(group_bits) };
    }

    assert!(
        _mm256_testz_si256(out_of_bounds, out_of_bounds) != 0,
        "index out of bounds"
    );

    let tail = total % 8;
    if tail == 0 {
        return BitBuffer::new(out.freeze(), total);
    }

    let start = full_groups * 8;
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
