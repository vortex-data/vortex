// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte-level compress using a `1 << 8 = 256`-entry lookup table.
//!
//! For each byte of the mask (8 bits -> 8 source elements), a precomputed permutation table
//! compacts the selected elements in a single indexed copy, without materializing indices or
//! slices.

use std::mem::MaybeUninit;

use super::bits::MaskBits;
use super::bits::for_each_mask_word;

/// For each mask byte (0..256), stores the element indices to keep and the count.
///
/// `BYTE_COMPRESS_LUT[mask_byte]` = `([i0, i1, ..., i7], popcount)` where
/// `i0..i_{popcount-1}` are the positions of set bits in `mask_byte`.
///
/// Total size: 256 * 9 = 2304 bytes, which trivially fits in L1 cache.
static BYTE_COMPRESS_LUT: &[([u8; 8], u8); 256] = &{
    let mut lut = [([0u8; 8], 0u8); 256];
    let mut mask: usize = 0;
    while mask < 256 {
        let mut indices = [0u8; 8];
        let mut count: u8 = 0;
        let mut bit: u8 = 0;
        while bit < 8 {
            if mask & (1 << bit) != 0 {
                indices[count as usize] = bit;
                count += 1;
            }
            bit += 1;
        }
        lut[mask] = (indices, count);
        mask += 1;
    }
    lut
};

/// Copy the elements of `src` selected by `bits` to `dst` and return the number copied.
///
/// # Safety
///
/// `bits` must select only elements of `src`, and `dst` must hold at least one value for each
/// selected element.
pub(super) unsafe fn compress<T: Copy>(
    src: &[T],
    bits: MaskBits<'_>,
    dst: &mut [MaybeUninit<T>],
) -> usize {
    let mut written = 0;

    // The word walk calls this closure from three places. Without `inline(always)`, LLVM keeps
    // it out of line in large callers such as `filter_records`, and sparse masks then pay for
    // one call per word.
    for_each_mask_word(
        bits,
        #[inline(always)]
        |word, word_start, word_len| {
            let (chunks, tail) = src[word_start..][..word_len].as_chunks::<8>();
            let mask_bytes = word.to_le_bytes();

            for (chunk, &mask_byte) in chunks.iter().zip(&mask_bytes) {
                // SAFETY: the mask byte selects only elements of `chunk`, and `dst` has room for
                // every selected element.
                written +=
                    unsafe { compress_byte(chunk, mask_byte, dst.get_unchecked_mut(written..)) };
            }

            if !tail.is_empty() {
                // SAFETY: the bits past `word_len` are cleared, so the mask byte selects only
                // elements of `tail`.
                written += unsafe {
                    compress_byte(
                        tail,
                        mask_bytes[chunks.len()],
                        dst.get_unchecked_mut(written..),
                    )
                };
            }
        },
    );

    written
}

/// Copy the elements of `chunk` selected by `mask_byte` to `dst` and return the number copied.
///
/// # Safety
///
/// `mask_byte` must select only elements of `chunk`, and `dst` must hold at least one value for
/// each selected element.
#[inline]
unsafe fn compress_byte<T: Copy>(chunk: &[T], mask_byte: u8, dst: &mut [MaybeUninit<T>]) -> usize {
    if mask_byte == 0 {
        return 0;
    }

    if chunk.len() == 8 && mask_byte == 0xFF {
        // All 8 selected, so bulk copy.
        dst[..8].write_copy_of_slice(chunk);
        return 8;
    }

    let (perm, count) = &BYTE_COMPRESS_LUT[mask_byte as usize];
    let count = *count as usize;
    // SAFETY: perm indices are all below `chunk.len()`, and `dst` has room for `count` values.
    unsafe {
        for j in 0..count {
            dst.get_unchecked_mut(j)
                .write(*chunk.get_unchecked(*perm.get_unchecked(j) as usize));
        }
    }
    count
}
