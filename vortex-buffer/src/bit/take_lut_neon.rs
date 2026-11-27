// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take bits from small bitmaps with NEON table lookups.
//!
//! NEON has no gather, but TBL looks up 16 bytes from a table of up to 64 bytes. A u8 index
//! addresses at most 256 bits, which fit in one table of 32 bytes. A bitmap of at most 4096 bits
//! fits in eight tables of 64 bytes, and TBX keeps the byte that an earlier table found when the
//! index is out of the range of its own table.

use std::arch::aarch64::uint8x16_t;
use std::arch::aarch64::uint8x16x4_t;
use std::arch::aarch64::vandq_u8;
use std::arch::aarch64::vbslq_u8;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::veorq_u8;
use std::arch::aarch64::vgetq_lane_u64;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vld1q_u8_x2;
use std::arch::aarch64::vld1q_u8_x4;
use std::arch::aarch64::vld1q_u16;
use std::arch::aarch64::vmovn_high_u16;
use std::arch::aarch64::vmovn_u16;
use std::arch::aarch64::vpaddq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vqtbl2q_u8;
use std::arch::aarch64::vqtbl4q_u8;
use std::arch::aarch64::vqtbx4q_u8;
use std::arch::aarch64::vreinterpretq_u64_u8;
use std::arch::aarch64::vshrn_high_n_u16;
use std::arch::aarch64::vshrn_n_u16;
use std::arch::aarch64::vshrq_n_u8;
use std::arch::aarch64::vtstq_u8;
use std::slice::from_raw_parts;

use num_traits::AsPrimitive;

use super::collect_words;
use super::padded_table;
use crate::BitBuffer;
use crate::BitBufferView;

/// The byte `i` holds `1 << (i % 8)`, the mask of bit `i % 8` in a byte.
const BIT_MASKS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Takes the bits at `indices` with a table lookup, or returns `None` if the index type has no
/// table kernel.
///
/// The caller must make sure that every index is less than `bits.len()`.
pub(super) fn take<I: AsPrimitive<usize>>(
    bits: BitBufferView<'_>,
    indices: &[I],
) -> Option<BitBuffer> {
    if size_of::<I>() == size_of::<u8>() {
        // A u8 index addresses at most 256 bits.
        let table = padded_table::<32>(bits.slice(..bits.len().min(256)));

        // SAFETY: I has the size of a u8 and the caller checked that the indices are not negative.
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u8>(), indices.len()) };

        // SAFETY: every aarch64 target has NEON.
        return Some(unsafe { take_u8(&table, indices) });
    }

    if size_of::<I>() == size_of::<u16>() && bits.len() <= 4096 && fast_table_lookups() {
        let table = padded_table::<512>(bits);

        // SAFETY: I has the size of a u16 and the caller checked that the indices are not
        // negative.
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u16>(), indices.len()) };

        // SAFETY: every aarch64 target has NEON.
        return Some(unsafe { take_u16(&table, indices) });
    }

    None
}

/// TBX is slow on Neoverse V1 (Graviton3), the only common core with SVE but not SVE2.
fn fast_table_lookups() -> bool {
    !std::arch::is_aarch64_feature_detected!("sve")
        || std::arch::is_aarch64_feature_detected!("sve2")
}

/// Packs the top bit of the 64 bytes in `set`, which are all ones or all zeros, into a word.
#[inline]
#[target_feature(enable = "neon")]
fn pack(set: [uint8x16_t; 4]) -> u64 {
    // SAFETY: BIT_MASKS holds 16 bytes.
    let weights = unsafe { vld1q_u8(BIT_MASKS.as_ptr()) };

    // Each pairwise add merges the weighted bits of neighbouring bytes, so three rounds give one
    // byte for each eight bytes of the input, in order.
    let quarters = [
        vandq_u8(set[0], weights),
        vandq_u8(set[1], weights),
        vandq_u8(set[2], weights),
        vandq_u8(set[3], weights),
    ];
    let halves = (
        vpaddq_u8(quarters[0], quarters[1]),
        vpaddq_u8(quarters[2], quarters[3]),
    );
    let whole = vpaddq_u8(halves.0, halves.1);
    vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(whole, whole)))
}

#[target_feature(enable = "neon")]
fn take_u8(table: &[u8; 32], indices: &[u8]) -> BitBuffer {
    // SAFETY: the table holds 32 bytes, and BIT_MASKS 16.
    let (table, bit_masks) = unsafe { (vld1q_u8_x2(table.as_ptr()), vld1q_u8(BIT_MASKS.as_ptr())) };
    let bit_index_bits = vdupq_n_u8(0x07);

    collect_words(indices, |chunk| {
        let lookup = |offset: usize| {
            // SAFETY: the chunk holds 64 indices and offset is at most 48.
            let indices = unsafe { vld1q_u8(chunk.as_ptr().add(offset)) };
            let byte = vqtbl2q_u8(table, vshrq_n_u8::<3>(indices));
            let mask = vqtbl1q_u8(bit_masks, vandq_u8(indices, bit_index_bits));
            vtstq_u8(byte, mask)
        };
        pack([lookup(0), lookup(16), lookup(32), lookup(48)])
    })
}

#[target_feature(enable = "neon")]
fn take_u16(table: &[u8; 512], indices: &[u16]) -> BitBuffer {
    // SAFETY: the table holds 512 bytes, so every load of 64 bytes at a multiple of 64 below 512
    // is in bounds.
    let load = |offset: usize| unsafe { vld1q_u8_x4(table.as_ptr().add(offset)) };
    let tables: [uint8x16x4_t; 8] = [
        load(0),
        load(64),
        load(128),
        load(192),
        load(256),
        load(320),
        load(384),
        load(448),
    ];

    // SAFETY: BIT_MASKS holds 16 bytes.
    let bit_masks = unsafe { vld1q_u8(BIT_MASKS.as_ptr()) };
    let bit_index_bits = vdupq_n_u8(0x07);
    let table_bits = [vdupq_n_u8(0x40), vdupq_n_u8(0x80), vdupq_n_u8(0xC0)];
    let half_bit = vdupq_n_u8(0x08);

    collect_words(indices, |chunk| {
        let lookup = |offset: usize| {
            // SAFETY: the chunk holds 64 indices and offset is at most 48.
            let (low, high) = unsafe {
                (
                    vld1q_u16(chunk.as_ptr().add(offset)),
                    vld1q_u16(chunk.as_ptr().add(offset + 8)),
                )
            };

            // The byte index has 9 bits. Its low 8 bits select a byte in each half of the table,
            // and bit 8, which is bit 11 of the index, selects the half.
            let byte_index = vshrn_high_n_u16::<3>(vshrn_n_u16::<3>(low), high);
            let index_high = vshrn_high_n_u16::<8>(vshrn_n_u16::<8>(low), high);

            // Each table covers 64 bytes, which bits 6 and 7 of the byte index select. The XOR
            // clears these bits only for the indices of one table. The other indices stay at 64
            // or more, so TBX keeps the byte that an earlier table found.
            let mut first = vqtbl4q_u8(tables[0], byte_index);
            let mut second = vqtbl4q_u8(tables[4], byte_index);
            for (t, bits) in table_bits.iter().enumerate() {
                let index = veorq_u8(byte_index, *bits);
                first = vqtbx4q_u8(first, tables[t + 1], index);
                second = vqtbx4q_u8(second, tables[t + 5], index);
            }
            let byte = vbslq_u8(vtstq_u8(index_high, half_bit), second, first);

            let bit_index = vmovn_high_u16(vmovn_u16(low), high);
            let mask = vqtbl1q_u8(bit_masks, vandq_u8(bit_index, bit_index_bits));
            vtstq_u8(byte, mask)
        };
        pack([lookup(0), lookup(16), lookup(32), lookup(48)])
    })
}
