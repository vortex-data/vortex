// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar compaction of slices by bits, indices, or ranges, both out of place and in place.

use std::mem::MaybeUninit;

use super::bits::MaskBits;
use super::bits::for_each_mask_word;
use super::bits::low_bits_mask;

/// Copy the elements of `src` selected by `bits` to `dst` and return the number copied.
///
/// # Safety
///
/// `bits` must select only elements of `src`, and `dst` must hold at least one value for each
/// selected element.
#[inline]
pub(super) unsafe fn compact_by_bitmap<T: Copy>(
    src: &[T],
    bits: MaskBits<'_>,
    dst: &mut [MaybeUninit<T>],
) -> usize {
    let mut write_pos = 0;

    // See `byte_compress::compress` for why this closure is always inlined.
    for_each_mask_word(bits, #[inline(always)] |word, word_start, word_len| {
        let all_selected = low_bits_mask(word_len);
        debug_assert_eq!(word & !all_selected, 0);

        if word == all_selected {
            dst[write_pos..][..word_len].write_copy_of_slice(&src[word_start..][..word_len]);
            write_pos += word_len;
        } else {
            let mut selected = word;
            while selected != 0 {
                let index = word_start + selected.trailing_zeros() as usize;
                // SAFETY: set bits index into `src`, and `dst` has room for every selected
                // element.
                unsafe {
                    dst.get_unchecked_mut(write_pos)
                        .write(*src.get_unchecked(index))
                };
                write_pos += 1;
                selected &= selected - 1;
            }
        }
    });

    write_pos
}

/// Copy the elements of `src` selected by `bits` to `dst` one run of set bits at a time, and
/// return the number copied.
///
/// This is faster than [`compact_by_bitmap`] when the selected elements form long runs.
// Inlining this walk next to `compact_by_bitmap` makes the bit walk 1.5x slower on Apple M4.
#[inline(never)]
pub(super) fn compact_runs_by_bitmap<T: Copy>(
    src: &[T],
    bits: MaskBits<'_>,
    dst: &mut [MaybeUninit<T>],
) -> usize {
    let mut write_pos = 0;

    for_each_mask_word(bits, |mut word, word_start, _| {
        while word != 0 {
            let run_start = word.trailing_zeros() as usize;
            let run_len = (!(word >> run_start)).trailing_zeros() as usize;
            dst[write_pos..][..run_len]
                .write_copy_of_slice(&src[word_start + run_start..][..run_len]);
            write_pos += run_len;

            // Adding the lowest bit of the run carries through the run and clears it.
            word &= word.wrapping_add(1 << run_start);
        }
    });

    write_pos
}

/// Copy the elements of `src` at the strictly increasing `indices` to `dst`.
///
/// # Safety
///
/// Each index must be in bounds of `src`.
pub(super) unsafe fn gather_indices<T: Copy>(
    src: &[T],
    indices: &[usize],
    dst: &mut [MaybeUninit<T>],
) {
    for (dst, &index) in dst[..indices.len()].iter_mut().zip(indices) {
        // SAFETY: forwarded from the caller contract.
        dst.write(unsafe { *src.get_unchecked(index) });
    }
}

/// Copy the `(start, end)` ranges of `src` to `dst` and return the number of elements copied.
pub(super) fn copy_slices<T: Copy>(
    src: &[T],
    slices: &[(usize, usize)],
    dst: &mut [MaybeUninit<T>],
) -> usize {
    let mut write_pos = 0;

    for &(start, end) in slices {
        dst[write_pos..][..end - start].write_copy_of_slice(&src[start..end]);
        write_pos += end - start;
    }

    write_pos
}

/// Compact the elements of `values` selected by `bits` to its front, and return the new length.
pub(super) fn compact_in_place_by_bitmap<T: Copy>(values: &mut [T], bits: MaskBits<'_>) -> usize {
    let mut write_pos = 0;

    for_each_mask_word(bits, |word, word_start, word_len| {
        let all_selected = low_bits_mask(word_len);
        debug_assert_eq!(word & !all_selected, 0);

        if word == all_selected {
            if write_pos != word_start {
                values.copy_within(word_start..word_start + word_len, write_pos);
            }
            write_pos += word_len;
        } else {
            let mut selected = word;
            while selected != 0 {
                let index = word_start + selected.trailing_zeros() as usize;
                // SAFETY: set bits are limited to `word_len` and stable compaction guarantees
                // `write_pos <= index`.
                unsafe { *values.get_unchecked_mut(write_pos) = *values.get_unchecked(index) };
                write_pos += 1;
                selected &= selected - 1;
            }
        }
    });

    write_pos
}

/// Compact the elements of `values` at the strictly increasing `indices` to its front, and
/// return the new length.
///
/// # Safety
///
/// Each index must be in bounds of `values`.
pub(super) unsafe fn compact_in_place_by_indices<T: Copy>(
    values: &mut [T],
    indices: &[usize],
) -> usize {
    for (write_pos, &index) in indices.iter().enumerate() {
        // SAFETY: the index is in bounds, and strictly increasing indices guarantee
        // `write_pos <= index`.
        unsafe { *values.get_unchecked_mut(write_pos) = *values.get_unchecked(index) };
    }

    indices.len()
}

/// Compact the sorted, disjoint `(start, end)` ranges of `values` to its front, and return the
/// new length.
pub(super) fn compact_in_place_by_slices<T: Copy>(
    values: &mut [T],
    slices: &[(usize, usize)],
) -> usize {
    let mut write_pos = 0;

    for &(start, end) in slices {
        if write_pos != start {
            values.copy_within(start..end, write_pos);
        }
        write_pos += end - start;
    }

    write_pos
}
