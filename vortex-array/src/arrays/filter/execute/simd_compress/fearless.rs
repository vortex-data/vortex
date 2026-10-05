// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compact eight byte lanes using a portable 16-byte shuffle.

use std::ptr;

use fearless_simd::prelude::*;
use fearless_simd::u8x16;
use vortex_mask::MaskValues;

use super::super::slice::for_each_mask_word;
use super::super::slice::low_bits_mask;
use super::bulk_copy;
use super::compress_lut;
use super::compress_tail;

static IDX_LUT: [[u8; 16]; 256] = compress_lut::<256, 16>(8, 1);

/// # Safety
///
/// The pointer contract of the parent module's filter entry points must hold.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn compress_word<S: Simd, const IN_PLACE: bool>(
    simd: S,
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
        unsafe { bulk_copy::<IN_PLACE>(src, dst, word_start, word_len, write_pos, 1) };
        return write_pos + word_len;
    }

    let mut sub = 0;
    while sub + 8 <= word_len {
        let mask = ((word >> sub) & 0xff) as usize;
        // SAFETY: the chunk holds eight in-bounds bytes. Materializing the value ends
        // the source borrow before an overlapping in-place store.
        let bytes = unsafe { src.add(word_start + sub).cast::<[u8; 8]>().read_unaligned() };
        let chunk = u8x16::from_slice(
            simd,
            &[
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], 0,
                0, 0, 0, 0, 0, 0, 0,
            ],
        );
        let indices = u8x16::from_slice(simd, &IDX_LUT[mask]);
        let packed = chunk.swizzle_dyn(indices).to_array();
        // SAFETY: output has vector slack; an in-place store ends no later than the
        // source chunk just loaded. Later stores overwrite unselected trailing bytes.
        unsafe { ptr::copy_nonoverlapping(packed.as_ptr(), dst.add(write_pos), 8) };
        write_pos += mask.count_ones() as usize;
        sub += 8;
    }

    if sub < word_len {
        let bits = (word >> sub) & low_bits_mask(word_len - sub);
        // SAFETY: forwarded from the caller contract, with only in-bounds tail bits.
        write_pos =
            unsafe { compress_tail::<IN_PLACE>(src, dst, bits, word_start + sub, write_pos, 1) };
    }
    write_pos
}

/// # Safety
///
/// The pointer contract of the parent module's filter entry points must hold.
pub(super) unsafe fn compress_fearless_8<const IN_PLACE: bool>(
    src: *const u8,
    dst: *mut u8,
    mask: &MaskValues,
) -> usize {
    fearless_simd::dispatch!(fearless_simd::Level::new(), simd => {
        let mut write_pos = 0;
        for_each_mask_word(mask, |word, word_start, word_len| {
            // SAFETY: forwarded from the caller contract.
            write_pos = unsafe {
                compress_word::<_, IN_PLACE>(simd, src, dst, word, word_start, word_len, write_pos)
            };
        });
        write_pos
    })
}
