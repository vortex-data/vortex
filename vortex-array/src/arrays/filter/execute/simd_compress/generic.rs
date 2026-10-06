// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Portable `fearless_simd` compress kernels for 1- and 2-byte elements.
//!
//! Each sub-word of 8 lanes is compacted with one 16-byte `swizzle_dyn` driven by a byte-index
//! lookup table. Dispatch wraps the whole mask walk, so the per-word body inlines into one
//! target-feature function rather than being called once per mask word.

use std::ptr;

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;
use fearless_simd::u8x16;
use fearless_simd_macros::simd;
use vortex_mask::MaskValues;

use super::super::slice::for_each_mask_word;
use super::super::slice::low_bits_mask;
use super::bulk_copy;
use super::compress_lut;
use super::compress_tail;

static IDX_LUT_8: [[u8; 16]; 256] = compress_lut::<256, 16>(8, 1);
static IDX_LUT_16: [[u8; 16]; 256] = compress_lut::<256, 16>(8, 2);

/// Compact one mask word of `ELEM`-byte elements, `LANES` elements per shuffle.
///
/// # Safety
///
/// The pointer contract of [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
/// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
#[expect(
    clippy::cast_possible_truncation,
    reason = "deliberate submask narrowing"
)]
#[allow(clippy::too_many_arguments)]
#[simd]
unsafe fn compress_word<S: Simd, const IN_PLACE: bool, const ELEM: usize, const LANES: usize>(
    simd: S,
    idx_lut: &[[u8; 16]],
    src: *const u8,
    dst: *mut u8,
    word: u64,
    word_start: usize,
    word_len: usize,
    mut write_pos: usize,
) -> usize {
    if word == 0 {
        return write_pos;
    }
    if word == low_bits_mask(word_len) {
        // SAFETY: forwarded from the caller contract.
        unsafe { bulk_copy::<IN_PLACE>(src, dst, word_start, word_len, write_pos, ELEM) };
        return write_pos + word_len;
    }

    // Empty chunks still store garbage that the next chunk overwrites; branching here
    // regresses masks near the density crossover.
    let mut sub = 0;
    while sub + LANES <= word_len {
        let m = ((word >> sub) & low_bits_mask(LANES)) as usize;
        // SAFETY: the chunk holds `LANES` in-bounds source elements.
        let chunk_ptr = unsafe { src.add((word_start + sub) * ELEM) };
        // Materializing the bytes ends the source read before an overlapping in-place store.
        let bytes: [u8; 16] = if ELEM * LANES == 16 {
            // SAFETY: see above; the chunk is exactly 16 bytes.
            unsafe { chunk_ptr.cast::<[u8; 16]>().read_unaligned() }
        } else {
            // SAFETY: see above; the chunk is exactly 8 bytes.
            let half = unsafe { chunk_ptr.cast::<[u8; 8]>().read_unaligned() };
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&half);
            bytes
        };
        let packed = u8x16::from_slice(simd, &bytes)
            .swizzle_dyn(u8x16::from_slice(simd, &idx_lut[m]))
            .to_array();
        // SAFETY: out-of-place output has vector slack. In-place, the store ends within the
        // source chunk already loaded, and later stores overwrite trailing garbage.
        unsafe {
            ptr::copy_nonoverlapping(packed.as_ptr(), dst.add(write_pos * ELEM), ELEM * LANES)
        };
        write_pos += m.count_ones() as usize;
        sub += LANES;
    }

    if sub < word_len {
        let bits = (word >> sub) & low_bits_mask(word_len - sub);
        // SAFETY: forwarded from the caller contract.
        write_pos =
            unsafe { compress_tail::<IN_PLACE>(src, dst, bits, word_start + sub, write_pos, ELEM) };
    }

    write_pos
}

/// Generate a mask-walking entry point for one element width.
macro_rules! generic_compress_kernel {
    ($walk_fn:ident,elem_size: $elem_size:literal,lanes: $lanes:literal,idx_lut: $idx_lut:ident) => {
        /// # Safety
        ///
        /// The pointer contract of [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        pub(super) unsafe fn $walk_fn<const IN_PLACE: bool>(
            src: *const u8,
            dst: *mut u8,
            mask: &MaskValues,
        ) -> usize {
            dispatch!(Level::new(), simd => {
                let mut write_pos = 0;
                for_each_mask_word(mask, |word, word_start, word_len| {
                    // SAFETY: forwarded from the caller contract.
                    write_pos = unsafe {
                        compress_word::<_, IN_PLACE, $elem_size, $lanes>(
                            simd, &$idx_lut, src, dst, word, word_start, word_len, write_pos,
                        )
                    };
                });
                write_pos
            })
        }
    };
}

generic_compress_kernel!(compress_generic_8, elem_size: 1, lanes: 8, idx_lut: IDX_LUT_8);
generic_compress_kernel!(compress_generic_16, elem_size: 2, lanes: 8, idx_lut: IDX_LUT_16);
