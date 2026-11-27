// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take bits from small bitmaps with byte table lookups instead of gathers.
//!
//! A u8 index addresses at most 256 bits, which fit in a table of 32 bytes. A bitmap of at most
//! 4096 bits fits in eight tables of 64 bytes, and a lookup of 128 bytes from two of them is one
//! VPERMI2B with AVX-512 VBMI, or a TBL and a TBX on NEON. Four of these lookups and three selects
//! find the byte of 64 u16 indices.

use std::array;
use std::slice::from_raw_parts;

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::prelude::*;
use fearless_simd::u8x16;
use fearless_simd::u8x64;
use fearless_simd::u16x32;
use fearless_simd_macros::simd;
use num_traits::AsPrimitive;

use super::collect_words;
use super::padded_table;
use crate::BitBuffer;
use crate::BitBufferView;
use crate::byte_table::block_tables;
use crate::byte_table::dispatch_byte_lookup;
use crate::byte_table::lookup_32;

/// The byte `i` holds `1 << (i % 8)`, the mask of bit `i % 8` in a byte.
const BIT_MASKS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Takes the bits at `indices` with a table lookup, or returns `None` if the CPU or the index type
/// has no table kernel.
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

        return dispatch_byte_lookup!(simd => take_u8(simd, &table, indices));
    }

    if size_of::<I>() == size_of::<u16>() && bits.len() <= 4096 {
        let table = padded_table::<512>(bits);

        // SAFETY: I has the size of a u16 and the caller checked that the indices are not
        // negative.
        let indices = unsafe { from_raw_parts(indices.as_ptr().cast::<u16>(), indices.len()) };

        return take_u16_dispatch(&table, indices);
    }

    None
}

/// Runs the u16 kernel where a lookup of 128 bytes is fast: with AVX-512 VBMI, and with NEON
/// except on Neoverse V1. Without VBMI, the AVX2 gather is faster.
fn take_u16_dispatch(table: &[u8; 512], indices: &[u16]) -> Option<BitBuffer> {
    let level = Level::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(simd) = level.as_avx512() {
        return Some(take_u16(simd, table, indices));
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(simd) = level.as_neon()
        && fast_table_lookups()
    {
        return Some(take_u16(simd, table, indices));
    }

    let _ = (level, table, indices);
    None
}

/// TBX is slow on Neoverse V1 (Graviton3), the only common core with SVE but not SVE2.
#[cfg(target_arch = "aarch64")]
fn fast_table_lookups() -> bool {
    !std::arch::is_aarch64_feature_detected!("sve")
        || std::arch::is_aarch64_feature_detected!("sve2")
}

/// Returns the bit of each byte in `bytes` that the low 3 bits of `bit_index` select, as a word.
#[inline]
fn select_bits<S: Simd>(simd: S, bytes: u8x64<S>, bit_index: u8x64<S>) -> u64 {
    let bit_masks = u8x64::block_splat(u8x16::from_slice(simd, &BIT_MASKS));
    let mask = bit_masks.swizzle_dyn_within_blocks(bit_index & u8x64::splat(simd, 0x07));
    (bytes & mask).simd_eq(mask).to_bitmask()
}

#[simd]
fn take_u8<S: Simd>(simd: S, table: &[u8; 32], indices: &[u8]) -> BitBuffer {
    let (low, high) = block_tables(simd, table);

    collect_words(indices, |chunk| {
        let indices = u8x64::from_slice(simd, chunk);
        let bytes = lookup_32(simd, low, high, indices >> 3);
        select_bits(simd, bytes, indices)
    })
}

#[simd]
fn take_u16<S: Simd>(simd: S, table: &[u8; 512], indices: &[u16]) -> BitBuffer {
    let tables: [u8x64<S>; 8] =
        array::from_fn(|i| u8x64::from_slice(simd, &table[i * 64..(i + 1) * 64]));
    let zero = u8x64::splat(simd, 0);

    collect_words(indices, |chunk| {
        let (first, second) = chunk.split_at(32);
        let first = u16x32::from_slice(simd, first);
        let second = u16x32::from_slice(simd, second);

        // The byte index has 9 bits. Its low 7 bits select a byte from a pair of tables, and its
        // top two bits select the pair.
        let byte_index = simd.narrow_u16x32(first >> 3, second >> 3);
        let pair_index = byte_index & u8x64::splat(simd, 0x7F);
        let bit_7 = byte_index.simd_ge(u8x64::splat(simd, 0x80));
        let bit_8 = simd.narrow_u16x32(first >> 11, second >> 11).simd_gt(zero);

        let pairs: [u8x64<S>; 4] =
            array::from_fn(|i| tables[2 * i].concat_swizzle_dyn(tables[2 * i + 1], pair_index));
        let low_half = bit_7.select(pairs[1], pairs[0]);
        let high_half = bit_7.select(pairs[3], pairs[2]);
        let bytes = bit_8.select(high_half, low_half);

        select_bits(simd, bytes, simd.narrow_u16x32(first, second))
    })
}
