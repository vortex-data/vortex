// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;
use vortex_compute::filter::Filter;

use crate::pipeline::BitView;

impl<'a, T: Copy> Filter<BitView<'a>> for &'a mut [T] {
    type Output = ();

    fn filter(self, mask: &BitView<'a>) -> Self::Output {
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
    }
}
