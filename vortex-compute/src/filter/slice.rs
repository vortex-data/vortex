// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-place filter implementations over mutable slices.
//!
//! Note that there is no `slice` module in `vortex-buffer` because slices always require a copy
//! to filter them. Therefore, it's likely better to implement the filter against the actual
//! zero-copy container type e.g. Buffer.

// This is modeled after the constant with the equivalent name in arrow-rs.
pub(super) const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

use std::ptr;

use vortex_buffer::{BitView, Buffer, BufferMut};
use vortex_mask::{Mask, MaskIter};

use crate::filter::Filter;

impl<T: Copy> Filter<Mask> for &[T] {
    type Output = Buffer<T>;

    fn filter(self, selection_mask: &Mask) -> Buffer<T> {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the buffer length"
        );

        match selection_mask {
            Mask::AllTrue(_) => Buffer::<T>::copy_from(self),
            Mask::AllFalse(_) => Buffer::empty(),
            Mask::Values(v) => match v.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
                MaskIter::Indices(indices) => filter_indices(self, indices),
                MaskIter::Slices(slices) => {
                    filter_slices(self, selection_mask.true_count(), slices).freeze()
                }
            },
        }
    }
}

pub(super) fn filter_indices<T: Copy>(values: &[T], indices: &[usize]) -> Buffer<T> {
    Buffer::<T>::from_trusted_len_iter(indices.iter().map(|&idx| values[idx]))
}

pub(super) fn filter_slices<T>(
    values: &[T],
    output_len: usize,
    slices: &[(usize, usize)],
) -> BufferMut<T> {
    let mut out = BufferMut::<T>::with_capacity(output_len);
    for (start, end) in slices {
        out.extend_from_slice(&values[*start..*end]);
    }
    out
}

impl<const NB: usize, T: Copy> Filter<BitView<'_, NB>> for &[T] {
    type Output = Buffer<T>;

    fn filter(self, mask: &BitView<NB>) -> Self::Output {
        assert_eq!(
            self.len(),
            BitView::<NB>::N,
            "Mask length must equal the slice length"
        );

        let mut read_ptr = self.as_ptr();

        let mut write = BufferMut::<T>::with_capacity(mask.true_count());
        unsafe { write.set_len(mask.true_count()) };

        let mut write_ptr = write.as_mut_ptr();

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
                        ptr::copy_nonoverlapping(read_ptr, write_ptr, usize::BITS as usize);
                        read_ptr = read_ptr.add(usize::BITS as usize);
                        write_ptr = write_ptr.add(usize::BITS as usize);
                    }
                }
                _ => {
                    // Iterate the bits in a word, attempting to copy contiguous runs of values.
                    let mut read_pos = 0;
                    let mut write_pos = 0;
                    while word != 0 {
                        let tz = word.trailing_zeros();
                        if tz > 0 {
                            // shift off the trailing zeros since they are unselected.
                            // this advances the read head, but not the write head.
                            read_pos += tz;
                            word >>= tz;
                            continue;
                        }

                        // copy the next several values to our out pointer.
                        let extent = word.trailing_ones();
                        unsafe {
                            ptr::copy_nonoverlapping(
                                read_ptr.add(read_pos as usize),
                                write_ptr.add(write_pos as usize),
                                extent as usize,
                            );
                        }
                        // Advance the reader and writer by the number of values
                        // we just copied.
                        read_pos += extent;
                        write_pos += extent;

                        // shift off the low bits of the word so we can copy the next run.
                        word >>= extent;
                    }

                    unsafe {
                        read_ptr = read_ptr.add(usize::BITS as usize);
                        write_ptr = write_ptr.add(write_pos as usize);
                    };
                }
            }
        }

        write.freeze()
    }
}
