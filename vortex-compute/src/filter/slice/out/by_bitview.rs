// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use vortex_buffer::BitView;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use crate::filter::Filter;

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
