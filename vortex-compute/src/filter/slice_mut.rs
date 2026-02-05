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
use vortex_mask::MaskValues;

use crate::filter::Filter;

impl<T: Copy> Filter<Mask> for &mut [T] {
    type Output = Self;

    fn filter(self, selection: &Mask) -> Self::Output {
        // We delegate checking that the mask length is equal to self to the `MaskValues`
        // filter implementation below.

        match selection {
            Mask::AllTrue(_) => self,
            Mask::AllFalse(_) => &mut self[..0],
            Mask::Values(v) => self.filter(v.as_ref()),
        }
    }
}

impl<T: Copy> Filter<MaskValues> for &mut [T] {
    type Output = Self;

    fn filter(self, mask_values: &MaskValues) -> Self::Output {
        assert_eq!(
            self.len(),
            mask_values.len(),
            "Mask length must equal the slice length"
        );

        // We choose to _always_ use slices here because iterating over indices will have strictly
        // more loop iterations than slices (more branches), and the overhead over batched
        // `ptr::copy(len)` is not that high.
        self.filter(mask_values.slices())
    }
}

impl<T: Copy> Filter<[usize]> for &mut [T] {
    type Output = Self;

    /// Filters by indices.
    ///
    /// The caller should ensure that the indices are strictly increasing, otherwise the resulting
    /// buffer might have strange values.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds. With the additional constraint that the indices are
    /// strictly increasing, the length of the indices must be less than or equal to the length of
    /// `self`.
    fn filter(self, indices: &[usize]) -> Self::Output {
        let mut write_pos = 0;

        for &idx in indices {
            // SAFETY: indices should be within bounds and we're copying one element at a time.
            unsafe {
                ptr::copy_nonoverlapping(
                    self.as_ptr().add(idx),
                    self.as_mut_ptr().add(write_pos),
                    1,
                )
            };
            write_pos += 1;
        }

        &mut self[..write_pos]
    }
}

impl<T: Copy> Filter<[(usize, usize)]> for &mut [T] {
    type Output = Self;

    /// Filters by ranges of indices.
    ///
    /// The caller should ensure that the ranges are strictly increasing, otherwise the resulting
    /// buffer might have strange values.
    ///
    /// # Panics
    ///
    /// Panics if any range is out of bounds. With the additional constraint that the ranges are
    /// strictly increasing, the length of the `slices` array must be less than or equal to the
    /// length of `self`.
    fn filter(self, slices: &[(usize, usize)]) -> Self::Output {
        let mut write_pos = 0;

        // For each range in the selection, copy all of the elements to the current write position.
        for &(start, end) in slices {
            // Note that we could add an if statement here that checks `if start != write_pos`, but
            // it's probably better to just avoid the branch misprediction.
            let len = end - start;

            // SAFETY: Slices should be within bounds.
            unsafe {
                ptr::copy(
                    self.as_ptr().add(start),
                    self.as_mut_ptr().add(write_pos),
                    len,
                )
            };

            write_pos += len;
        }

        &mut self[..write_pos]
    }
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
                            ptr::copy(
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

        &mut self[..mask.true_count()]
    }
}
