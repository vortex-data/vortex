// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-place filter implementations over mutable slices.
//!
//! Note that there is no `slice` module in `vortex-buffer` because slices always require a copy
//! to filter them. Therefore, it's likely better to implement the filter against the actual
//! zero-copy container type e.g. Buffer.

use std::ptr;

use vortex_buffer::BitView;
use vortex_mask::Mask;

use crate::filter::Filter;

impl<T: Copy> Filter<Mask> for &mut [T] {
    type Output = Self;

    fn filter(self, selection: &Mask) -> Self::Output {
        assert_eq!(
            self.len(),
            selection.len(),
            "Mask length must equal the slice length"
        );
        match selection {
            Mask::AllTrue(_) => self,
            Mask::AllFalse(_) => &mut self[..0],
            Mask::Values(v) => {
                // We choose to _always_ use slices here because iterating over indices will have
                // strictly more loop iterations than slices, and the overhead over batched
                // `ptr::copy(len)` is not worth it.
                let slices = v.slices();

                // SAFETY: We checked above that the selection mask has the same length as the
                // buffer.
                unsafe { filter_slices_in_place(self, slices) }
            }
        }
    }
}

/// Filters a buffer in-place using slice ranges to determine which values to keep.
///
/// Returns the new length of the buffer.
///
/// # Safety
///
/// The slice ranges must be in the range of the `buffer`.
unsafe fn filter_slices_in_place<'a, T: Copy>(
    buffer: &'a mut [T],
    slices: &[(usize, usize)],
) -> &'a mut [T] {
    let mut write_pos = 0;

    // For each range in the selection, copy all of the elements to the current write position.
    for &(start, end) in slices {
        // Note that we could add an if statement here that checks `if read_idx != write_idx`, but
        // it's probably better to just avoid the branch misprediction.

        let len = end - start;

        // SAFETY: The safety contract enforces that all ranges are within bounds.
        unsafe {
            ptr::copy(
                buffer.as_ptr().add(start),
                buffer.as_mut_ptr().add(write_pos),
                len,
            )
        };

        write_pos += len;
    }

    &mut buffer[..write_pos]
}

impl<'a, const NB: usize, T: Copy> Filter<BitView<'a, NB>> for &mut [T] {
    type Output = Self;

    fn filter(self, mask: &BitView<'a, NB>) -> Self::Output {
        assert_eq!(
            self.len(),
            BitView::<NB>::N,
            "Mask length must equal the slice length"
        );

        let mut read_ptr = self.as_ptr();
        let mut write_ptr = self.as_mut_ptr();

        // First we loop 64 elements at a time (usize::BITS)
        for mut word in mask.iter_words() {
            match word {
                0usize => {
                    // No bits set => skip usize::BITS slice.
                    unsafe {
                        read_ptr = read_ptr.add(usize::BITS as usize);
                    }
                }
                usize::MAX => {
                    // All slice => copy usize::BITS slice.
                    unsafe {
                        // We cannot guarantee non-overlapping, so use ptr::copy rather than
                        // ptr::copy_nonoverlapping.
                        ptr::copy(read_ptr, write_ptr, usize::BITS as usize);
                        read_ptr = read_ptr.add(usize::BITS as usize);
                        write_ptr = write_ptr.add(usize::BITS as usize);
                    }
                }
                _ => {
                    while word != 0 {
                        let bit_pos = word.trailing_zeros();
                        word &= word - 1; // Clear the bit at `bit_pos`
                        let span = word.trailing_ones();
                        word >>= span;
                        unsafe {
                            ptr::copy(read_ptr.add(bit_pos as usize), write_ptr, span as usize);
                            write_ptr = write_ptr.add(span as usize);
                        }
                    }
                    unsafe { read_ptr = read_ptr.add(usize::BITS as usize) };
                }
            }
        }

        &mut self[..mask.true_count()]
    }
}
