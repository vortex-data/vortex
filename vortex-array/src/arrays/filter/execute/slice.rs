// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core slice-level filtering algorithms.
//!
//! Provides both immutable and mutable (in-place) filtering of typed slices by cached mask
//! representations or directly from the mask bitmap.

use std::mem::MaybeUninit;

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_mask::MaskValues;

/// The mask bits that a word walker visits.
#[derive(Clone, Copy)]
pub(super) enum MaskBits<'a> {
    /// The whole bitmap of a mask.
    Mask(&'a MaskValues),
    /// The first `len` bits of `words`, with any bits past `len` cleared.
    Words { words: &'a [u64], len: usize },
}

/// Invoke `f` with each `(word, word_start, word_len)` of `bits`, where `word` holds the mask
/// bits for elements `word_start..word_start + word_len` in its low `word_len` bits.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn for_each_mask_word(bits: MaskBits<'_>, mut f: impl FnMut(u64, usize, usize)) {
    let mask = match bits {
        MaskBits::Mask(mask) => mask,
        MaskBits::Words { words, len } => {
            debug_assert!(words.len() * 64 >= len);
            for (word_idx, &word) in words.iter().enumerate() {
                let word_start = word_idx * 64;
                f(word, word_start, (len - word_start).min(64));
            }
            return;
        }
    };

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

/// A `u64` with the low `len` bits set.
#[inline]
pub(super) fn low_bits_mask(len: usize) -> u64 {
    debug_assert!(len <= 64);
    if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

/// Filter a slice from the mask bitmap without materializing indices or ranges.
pub(super) fn filter_slice_by_bitmap<T: Copy>(
    slice: &[T],
    mask: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    assert_eq!(
        mask.len(),
        slice.len(),
        "Selection mask length must equal the buffer length"
    );

    let output_len = mask.true_count();
    let mut out = BufferMut::<T>::with_capacity_in(output_len, allocator.clone());
    // SAFETY: the mask selects only elements of `slice`, and the output was allocated for exactly
    // `mask.true_count()` values.
    let written =
        unsafe { compact_by_bitmap(slice, MaskBits::Mask(mask), out.spare_capacity_mut()) };

    debug_assert_eq!(written, output_len);
    // SAFETY: every output slot was initialized exactly once above.
    unsafe { out.set_len(output_len) };
    out.freeze()
}

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

    for_each_mask_word(bits, |word, word_start, word_len| {
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

/// Filter a slice by a set of strictly increasing indices.
pub(super) fn filter_slice_by_indices<T: Copy>(
    slice: &[T],
    indices: &[usize],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());

    for (dst, &index) in out.spare_capacity_mut().iter_mut().zip(indices) {
        // SAFETY: mask indices are validated when the mask is constructed.
        dst.write(unsafe { *slice.get_unchecked(index) });
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
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity_in(output_len, allocator.clone());
    for (start, end) in slices {
        out.extend_from_slice(&slice[*start..*end]);
    }

    out.freeze()
}

/// Filter a mutable slice in-place from the mask bitmap, returning the new valid length.
pub(super) fn filter_slice_mut_by_bitmap<T: Copy>(slice: &mut [T], mask: &MaskValues) -> usize {
    assert_eq!(
        slice.len(),
        mask.len(),
        "Mask length must equal the slice length"
    );

    let mut write_pos = 0;

    for_each_mask_word(MaskBits::Mask(mask), |word, word_start, word_len| {
        let all_selected = low_bits_mask(word_len);
        debug_assert_eq!(word & !all_selected, 0);
        if word == all_selected {
            if write_pos != word_start {
                slice.copy_within(word_start..word_start + word_len, write_pos);
            }
            write_pos += word_len;
        } else {
            let mut selected = word;
            while selected != 0 {
                let index = word_start + selected.trailing_zeros() as usize;
                // SAFETY: set bits are limited to `word_len` and stable compaction guarantees
                // `write_pos <= index`.
                unsafe { *slice.get_unchecked_mut(write_pos) = *slice.get_unchecked(index) };
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
    for (write_pos, &index) in indices.iter().enumerate() {
        // SAFETY: mask indices are in bounds and stable compaction guarantees
        // `write_pos <= index`.
        unsafe { *slice.get_unchecked_mut(write_pos) = *slice.get_unchecked(index) };
    }
    indices.len()
}

/// Filter a mutable slice in-place by a set of `(start, end)` ranges, returning the new length.
pub(super) fn filter_slice_mut_by_slices<T: Copy>(
    slice: &mut [T],
    slices: &[(usize, usize)],
) -> usize {
    let mut write_pos = 0;

    for &(start, end) in slices {
        let len = end - start;

        if write_pos != start {
            slice.copy_within(start..end, write_pos);
        }

        write_pos += len;
    }

    write_pos
}
