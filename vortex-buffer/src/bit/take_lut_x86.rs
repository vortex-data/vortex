// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take bits from small bitmaps with byte table lookups instead of gathers.
//!
//! A bitmap of at most 4096 bits fits in eight AVX-512 registers. VPERMI2B looks up 64 bytes from
//! a 128-byte table in one instruction, so four of them and three blends find the byte of 64
//! indices. A u8 index can only address 256 bits, which fit in two 16-byte tables for VPSHUFB.

use std::arch::x86_64::__m256i;
use std::arch::x86_64::__m512i;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm256_and_si256;
use std::arch::x86_64::_mm256_blendv_epi8;
use std::arch::x86_64::_mm256_broadcastsi128_si256;
use std::arch::x86_64::_mm256_cmpeq_epi8;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_movemask_epi8;
use std::arch::x86_64::_mm256_set1_epi8;
use std::arch::x86_64::_mm256_shuffle_epi8;
use std::arch::x86_64::_mm256_slli_epi16;
use std::arch::x86_64::_mm256_srli_epi16;
use std::arch::x86_64::_mm512_and_si512;
use std::arch::x86_64::_mm512_broadcast_i32x4;
use std::arch::x86_64::_mm512_loadu_si512;
use std::arch::x86_64::_mm512_mask_blend_epi8;
use std::arch::x86_64::_mm512_movepi8_mask;
use std::arch::x86_64::_mm512_permutex2var_epi8;
use std::arch::x86_64::_mm512_set1_epi8;
use std::arch::x86_64::_mm512_shuffle_epi8;
use std::arch::x86_64::_mm512_srli_epi16;
use std::arch::x86_64::_mm512_test_epi8_mask;
use std::slice::from_raw_parts;

use num_traits::AsPrimitive;

use super::collect_words;
use super::padded_table;
use crate::BitBuffer;
use crate::BitBufferView;

/// The byte `i` holds `1 << (i % 8)`, the mask of bit `i % 8` in a byte.
const BIT_MASKS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Selects the even bytes of two registers: the low bytes of their u16 lanes.
const LOW_BYTES: [u8; 64] = {
    let mut lanes = [0u8; 64];
    let mut i = 0u8;
    while i < 64 {
        lanes[i as usize] = 2 * i;
        i += 1;
    }
    lanes
};

/// Selects the odd bytes of two registers: the high bytes of their u16 lanes.
const HIGH_BYTES: [u8; 64] = {
    let mut lanes = [0u8; 64];
    let mut i = 0u8;
    while i < 64 {
        lanes[i as usize] = 2 * i + 1;
        i += 1;
    }
    lanes
};

/// Takes the bits at `indices` with a table lookup, or returns `None` if the CPU or the index type
/// has no table kernel.
///
/// The caller must make sure that every index is less than `bits.len()`.
pub(super) fn take<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
) -> Option<BitBuffer> {
    if size_of::<I>() == size_of::<u8>() {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }

        // A u8 index addresses at most 256 bits.
        let table = padded_table::<32>(bits.slice(..bits.len().min(256)));

        // SAFETY: I has the size of a u8 and the caller checked that the indices are not negative.
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u8>(), indices.len()) };

        // SAFETY: we checked for AVX2 above.
        return Some(unsafe { take_u8_avx2(&table, indices) });
    }

    if size_of::<I>() == size_of::<u16>() && bits.len() <= 4096 {
        if !(is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi"))
        {
            return None;
        }

        let table = padded_table::<512>(bits);

        // SAFETY: I has the size of a u16 and the caller checked that the indices are not
        // negative.
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u16>(), indices.len()) };

        // SAFETY: we checked for AVX-512 VBMI above.
        return Some(unsafe { take_u16_vbmi(&table, indices) });
    }

    None
}

#[target_feature(enable = "avx2")]
fn take_u8_avx2(table: &[u8; 32], indices: &[u8]) -> BitBuffer {
    // SAFETY: the table holds 32 bytes, and BIT_MASKS 16.
    let (low_table, high_table, bit_masks) = unsafe {
        (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().add(16).cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(BIT_MASKS.as_ptr().cast())),
        )
    };
    let byte_index_bits = _mm256_set1_epi8(0x1F);
    let bit_index_bits = _mm256_set1_epi8(0x07);

    let lookup = |indices: __m256i| -> u32 {
        // The shift crosses into the next byte of the u16 lane, so mask the 5 bits that remain.
        let byte_index = _mm256_and_si256(_mm256_srli_epi16::<3>(indices), byte_index_bits);

        // VPSHUFB uses the low 4 bits of each index. Bit 4 selects the table with the blend,
        // which reads the top bit of each byte. The shift stays inside each byte, because no
        // byte index is more than 31.
        let low = _mm256_shuffle_epi8(low_table, byte_index);
        let high = _mm256_shuffle_epi8(high_table, byte_index);
        let byte = _mm256_blendv_epi8(low, high, _mm256_slli_epi16::<3>(byte_index));

        let mask = _mm256_shuffle_epi8(bit_masks, _mm256_and_si256(indices, bit_index_bits));
        let set = _mm256_cmpeq_epi8(_mm256_and_si256(byte, mask), mask);
        _mm256_movemask_epi8(set).cast_unsigned()
    };

    collect_words(indices, |chunk| {
        // SAFETY: the chunk holds 64 indices.
        let (first, second) = unsafe {
            (
                _mm256_loadu_si256(chunk.as_ptr().cast()),
                _mm256_loadu_si256(chunk.as_ptr().add(32).cast()),
            )
        };
        u64::from(lookup(first)) | (u64::from(lookup(second)) << 32)
    })
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn take_u16_vbmi(table: &[u8; 512], indices: &[u16]) -> BitBuffer {
    // SAFETY: the table holds 512 bytes, so every load of 64 bytes at a multiple of 64 below 512
    // is in bounds.
    let load = |offset: usize| unsafe { _mm512_loadu_si512(table.as_ptr().add(offset).cast()) };

    // SAFETY: the constants hold 64 or 16 bytes.
    let (tables, low_bytes, high_bytes, bit_masks) = unsafe {
        (
            [
                load(0),
                load(64),
                load(128),
                load(192),
                load(256),
                load(320),
                load(384),
                load(448),
            ],
            _mm512_loadu_si512(LOW_BYTES.as_ptr().cast()),
            _mm512_loadu_si512(HIGH_BYTES.as_ptr().cast()),
            _mm512_broadcast_i32x4(_mm_loadu_si128(BIT_MASKS.as_ptr().cast())),
        )
    };
    let bit_index_bits = _mm512_set1_epi8(0x07);

    collect_words(indices, |chunk| {
        // SAFETY: the chunk holds 64 indices of 2 bytes each.
        let (first, second): (__m512i, __m512i) = unsafe {
            (
                _mm512_loadu_si512(chunk.as_ptr().cast()),
                _mm512_loadu_si512(chunk.as_ptr().add(32).cast()),
            )
        };

        // The byte index has 9 bits. Its low byte goes to VPERMI2B, which reads 7 bits, and its
        // top two bits select one of the four lookups.
        let first_bytes = _mm512_srli_epi16::<3>(first);
        let second_bytes = _mm512_srli_epi16::<3>(second);
        let byte_index = _mm512_permutex2var_epi8(first_bytes, low_bytes, second_bytes);
        let byte_index_high = _mm512_permutex2var_epi8(first_bytes, high_bytes, second_bytes);
        let bit_7 = _mm512_movepi8_mask(byte_index);
        let bit_8 = _mm512_test_epi8_mask(byte_index_high, byte_index_high);

        let quarter_0 = _mm512_permutex2var_epi8(tables[0], byte_index, tables[1]);
        let quarter_1 = _mm512_permutex2var_epi8(tables[2], byte_index, tables[3]);
        let quarter_2 = _mm512_permutex2var_epi8(tables[4], byte_index, tables[5]);
        let quarter_3 = _mm512_permutex2var_epi8(tables[6], byte_index, tables[7]);
        let half_0 = _mm512_mask_blend_epi8(bit_7, quarter_0, quarter_1);
        let half_1 = _mm512_mask_blend_epi8(bit_7, quarter_2, quarter_3);
        let byte = _mm512_mask_blend_epi8(bit_8, half_0, half_1);

        let bit_index = _mm512_permutex2var_epi8(first, low_bytes, second);
        let mask = _mm512_shuffle_epi8(bit_masks, _mm512_and_si512(bit_index, bit_index_bits));
        _mm512_test_epi8_mask(byte, mask)
    })
}
