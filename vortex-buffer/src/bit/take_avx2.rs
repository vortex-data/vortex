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
        // SAFETY: i16 can be reinterpreted as u16 since we guarantee no
        // negative indices
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
            return Some(take_group(bits, indices, validity, last_dword, group));
        }
    }
    if size == size_of::<u32>() {
        // SAFETY: i32 can be reinterpreted as u32 since we guarantee
        // no negative indices
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u32>(), indices.len()) };
        let indices_ptr: *const u32 = indices.as_ptr();
        unsafe {
            let group = |group_idx| -> __m256i {
                let group_ptr: *const __m256i = indices_ptr.add(group_idx * 8).cast();
                // copy 256 bits into __m256i
                _mm256_loadu_si256(group_ptr)
            };
            return Some(take_group(bits, indices, validity, last_dword, group));
        }
    }
    None
}

#[target_feature(enable = "avx2")]
#[expect(clippy::cast_possible_truncation)]
unsafe fn take_group<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
    validity: Option<BitBufferView<'_>>,
    last_dword: usize,
    load_group: impl Fn(usize) -> __m256i,
) -> BitBuffer {
    let inner = bits.inner();
    let base = inner.as_ptr();
    let zero: __m256i = _mm256_setzero_si256();

    let Some(validity) = validity else {
        let gather = |_group: usize, dword: __m256i| -> (__m256i, __m256i) {
            let gathered = unsafe { _mm256_i32gather_epi32::<4>(base.cast(), dword) };
            (gathered, zero)
        };
        return unsafe { take_lanes(bits, indices, load_group, gather, None) };
    };

    let valid_bytes = validity.inner();
    let valid_offset = validity.offset();
    let valid_shift = valid_offset % 8;
    let valid_byte0 = valid_offset / 8;
    let full_groups = indices.len() / 8;
    if full_groups > 0 {
        let last_byte = valid_byte0 + (full_groups - 1) + usize::from(valid_shift != 0);
        assert!(last_byte < valid_bytes.len(), "validity out of bounds");
    }

    let max_dword: __m256i = _mm256_set1_epi32(last_dword as i32);
    let lane_bits: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    let gather = |group: usize, dword: __m256i| -> (__m256i, __m256i) {
        let byte = valid_byte0 + group;
        // SAFETY: last_byte was checked above
        let valid = unsafe {
            if valid_shift == 0 {
                *valid_bytes.get_unchecked(byte)
            } else {
                let lo = u16::from(*valid_bytes.get_unchecked(byte));
                let hi = u16::from(*valid_bytes.get_unchecked(byte + 1));
                let window = lo | (hi << 8);
                ((window >> valid_shift) & 0xFF) as u8
            }
        };
        let broadcast = _mm256_set1_epi32(i32::from(valid));
        let selected = _mm256_and_si256(broadcast, lane_bits);
        let lanes = _mm256_cmpeq_epi32(selected, lane_bits);
        let clamped = _mm256_min_epu32(dword, max_dword);
        let in_range = _mm256_cmpeq_epi32(clamped, dword);
        let escaped = _mm256_andnot_si256(in_range, lanes);
        let gathered =
            unsafe { _mm256_mask_i32gather_epi32::<4>(zero, base.cast(), clamped, lanes) };
        (gathered, escaped)
    };
    unsafe {
        take_lanes(
            bits,
            indices,
            load_group,
            gather,
            Some((valid_bytes, valid_offset)),
        )
    }
}

#[expect(clippy::cast_possible_truncation)]
#[target_feature(enable = "avx2")]
unsafe fn take_lanes<I>(
    bits: BitBufferView<'_>,
    indices: &[I],
    load_group: impl Fn(usize) -> __m256i,
    gather: impl Fn(usize, __m256i) -> (__m256i, __m256i),
    validity: Option<(&[u8], usize)>,
) -> BitBuffer
where
    I: AsPrimitive<usize>,
{
    let total = indices.len();
    let full_groups = total / 8;

    let bit_offset: __m256i = _mm256_set1_epi32(bits.offset() as i32);
    let low_bits: __m256i = _mm256_set1_epi32(31);
    let mut out_of_bounds: __m256i = _mm256_setzero_si256();

    let mut out = BufferMut::<u8>::with_capacity(total.div_ceil(8));
    for group in 0..full_groups {
        let bitpos = _mm256_add_epi32(load_group(group), bit_offset);
        let dword = _mm256_srli_epi32::<5>(bitpos);
        let (gathered, escaped) = gather(group, dword);
        out_of_bounds = _mm256_or_si256(out_of_bounds, escaped);
        let shift = _mm256_and_si256(bitpos, low_bits);
        let shifted = _mm256_srlv_epi32(gathered, shift);
        let top = _mm256_slli_epi32::<31>(shifted);
        let as_ps = _mm256_castsi256_ps(top);
        let group_bits = _mm256_movemask_ps(as_ps);
        // SAFETY: out has sufficient capacity
        unsafe { out.push_unchecked((group_bits & 0xFF) as u8) };
    }

    assert!(
        _mm256_testz_si256(out_of_bounds, out_of_bounds) != 0,
        "take index out of bounds"
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
        let keep = validity.map_or(usize::MAX, |(valid_bytes, valid_offset)| {
            // SAFETY: validity has the same length as indices
            (unsafe { get_bit_unchecked(valid_bytes.as_ptr(), valid_offset + pos) } as usize)
                .wrapping_neg()
        });
        // SAFETY: pos stays within indices for the tail range
        let idx = unsafe { indices.get_unchecked(pos) }.as_() & keep;
        let value = get_bit(inner, bits.offset() + idx);
        byte |= (value as u8) << bit;
    }
    // SAFETY: out was reserved for the tail byte
    unsafe { out.push_unchecked(byte) }
    BitBuffer::new(out.freeze(), total)
}
