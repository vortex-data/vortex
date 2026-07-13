// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core slice-level filtering algorithms.
//!
//! Provides both immutable and mutable (in-place) filtering of typed slices by cached mask
//! representations or directly from the mask bitmap.

use std::ptr;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::MaskValues;

#[inline]
fn for_each_mask_word(mask: &MaskValues, mut f: impl FnMut(u64, usize, usize)) {
    let bits = mask.bit_buffer();
    let unaligned = bits.unaligned_chunks();
    let lead = unaligned.lead_padding();
    let mut base = 0;

    if let Some(prefix) = unaligned.prefix() {
        let len = (64 - lead).min(mask.len());
        f(prefix >> lead, base, len);
        base += len;
    }

    for &word in unaligned.chunks() {
        f(word, base, 64);
        base += 64;
    }

    if let Some(suffix) = unaligned.suffix() {
        let len = mask.len() - base;
        f(suffix, base, len);
        base += len;
    }

    debug_assert_eq!(base, mask.len());
}

#[inline]
fn low_bits_mask(len: usize) -> u64 {
    debug_assert!(len <= 64);
    if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

// ---------------------------------------------------------------------------
// Immutable slice filtering
// ---------------------------------------------------------------------------

/// Filter a slice from the mask bitmap without materializing indices or ranges.
pub(super) fn filter_slice_by_bitmap<T: Copy>(slice: &[T], mask: &MaskValues) -> Buffer<T> {
    assert_eq!(
        mask.len(),
        slice.len(),
        "Selection mask length must equal the buffer length"
    );

    let output_len = mask.true_count();
    let mut out = BufferMut::<T>::with_capacity(output_len);
    let src_ptr = slice.as_ptr();
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    let mut write_pos = 0;

    for_each_mask_word(mask, |word, word_start, word_len| {
        let all_selected = low_bits_mask(word_len);
        debug_assert_eq!(word & !all_selected, 0);
        if word == all_selected {
            // SAFETY: a full mask word selects `word_len` in-bounds source values and the output
            // was allocated for every selected value.
            unsafe {
                ptr::copy_nonoverlapping(src_ptr.add(word_start), out_ptr.add(write_pos), word_len);
            }
            write_pos += word_len;
        } else {
            let mut selected = word;
            while selected != 0 {
                let index = word_start + selected.trailing_zeros() as usize;
                // SAFETY: set bits are limited to `word_len`, and the output was allocated for
                // exactly `mask.true_count()` values.
                unsafe {
                    out_ptr.add(write_pos).write(*src_ptr.add(index));
                }
                write_pos += 1;
                selected &= selected - 1;
            }
        }
    });

    debug_assert_eq!(write_pos, output_len);
    // SAFETY: every output slot was initialized exactly once above.
    unsafe { out.set_len(output_len) };
    out.freeze()
}

/// Filter a slice by a set of strictly increasing indices.
pub(super) fn filter_slice_by_indices<T: Copy>(slice: &[T], indices: &[usize]) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity(indices.len());
    let src_ptr = slice.as_ptr();
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for (write_pos, &index) in indices.iter().enumerate() {
        // SAFETY: mask indices are validated when the mask is constructed and the output has one
        // slot allocated for every index.
        unsafe { out_ptr.add(write_pos).write(*src_ptr.add(index)) };
    }

    // SAFETY: the loop initialized every output slot.
    unsafe { out.set_len(indices.len()) };
    out.freeze()
}

/// Filter a slice by a set of strictly increasing `(start, end)` ranges.
pub(super) fn filter_slice_by_slices<T: Copy>(
    slice: &[T],
    slices: &[(usize, usize)],
    output_len: usize,
) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity(output_len);
    for (start, end) in slices {
        out.extend_from_slice(&slice[*start..*end]);
    }

    out.freeze()
}

// ---------------------------------------------------------------------------
// Mutable (in-place) slice filtering
// ---------------------------------------------------------------------------

/// Filter a mutable slice in-place from the mask bitmap, returning the new valid length.
pub(super) fn filter_slice_mut_by_bitmap<T: Copy>(slice: &mut [T], mask: &MaskValues) -> usize {
    assert_eq!(
        slice.len(),
        mask.len(),
        "Mask length must equal the slice length"
    );

    let ptr = slice.as_mut_ptr();
    let mut write_pos = 0;

    for_each_mask_word(mask, |word, word_start, word_len| {
        let all_selected = low_bits_mask(word_len);
        debug_assert_eq!(word & !all_selected, 0);
        if word == all_selected {
            if write_pos != word_start {
                // SAFETY: source and destination are in bounds and may overlap while compacting
                // toward the start of the same allocation.
                unsafe { ptr::copy(ptr.add(word_start), ptr.add(write_pos), word_len) };
            }
            write_pos += word_len;
        } else {
            let mut selected = word;
            while selected != 0 {
                let index = word_start + selected.trailing_zeros() as usize;
                if write_pos != index {
                    // SAFETY: set bits are limited to `word_len` and stable compaction guarantees
                    // `write_pos <= index`.
                    unsafe { ptr::copy(ptr.add(index), ptr.add(write_pos), 1) };
                }
                write_pos += 1;
                selected &= selected - 1;
            }
        }
    });

    debug_assert_eq!(write_pos, mask.true_count());
    write_pos
}

/// Filter a mutable slice in-place by strictly increasing indices.
pub(super) fn filter_slice_mut_by_indices<T: Copy>(slice: &mut [T], indices: &[usize]) -> usize {
    let ptr = slice.as_mut_ptr();
    for (write_pos, &index) in indices.iter().enumerate() {
        if write_pos != index {
            // SAFETY: mask indices are in bounds and stable compaction guarantees
            // `write_pos <= index`.
            unsafe { ptr::copy(ptr.add(index), ptr.add(write_pos), 1) };
        }
    }
    indices.len()
}

/// Filter a mutable slice in-place by a set of `(start, end)` ranges, returning the new length.
pub(super) fn filter_slice_mut_by_slices<T: Copy>(
    slice: &mut [T],
    slices: &[(usize, usize)],
) -> usize {
    let mut write_pos = 0;

    // For each range in the selection, copy all of the elements to the current write position.
    for &(start, end) in slices {
        let len = end - start;

        if write_pos != start {
            // SAFETY: mask slices are in bounds and source and destination may overlap while
            // compacting toward the start of the same allocation.
            unsafe {
                ptr::copy(
                    slice.as_ptr().add(start),
                    slice.as_mut_ptr().add(write_pos),
                    len,
                )
            };
        }

        write_pos += len;
    }

    write_pos
}
