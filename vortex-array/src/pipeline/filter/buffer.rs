// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::bit_view::BitView;
use crate::pipeline::N;
use vortex_compute::filter::Filter;

impl<'a, T: Copy> Filter<BitView<'a>> for &'a mut [T] {
    type Output = ();

    fn filter(self, selection: &BitView<'a>) -> Self::Output {
        match selection.true_count() {
            0 => {
                // If the mask has no true bits, we set the length to 0.
            }
            N => {
                // If the mask has N true bits, we copy all elements.
            }
            n if n > 3 * N / 4 => {
                // High density: use iter_zeros to compact by removing gaps
                let mut write_idx = 0;
                let mut read_idx = 0;

                selection.iter_zeros(|zero_idx| {
                    // Copy elements from read_idx to zero_idx (exclusive) to write_idx
                    let count = zero_idx - read_idx;
                    unsafe {
                        // SAFETY: We assume that the elements are of type E and that the view is valid.
                        // Using memmove for potentially overlapping regions
                        std::ptr::copy(
                            self.as_ptr().add(read_idx),
                            self.as_mut_ptr().add(write_idx),
                            count,
                        );
                        write_idx += count;
                    }
                    read_idx = zero_idx + 1;
                });

                // Copy any remaining elements after the last zero
                unsafe {
                    std::ptr::copy(
                        self.as_ptr().add(read_idx),
                        self.as_mut_ptr().add(write_idx),
                        N - read_idx,
                    );
                }
            }
            _ => {
                let mut offset = 0;
                selection.iter_ones(|idx| {
                    unsafe {
                        // SAFETY: We assume that the elements are of type E and that the view is valid.
                        let value = *self.get_unchecked(idx);
                        // TODO(joe): use ptr increment (not offset).
                        *self.get_unchecked_mut(offset) = value;

                        offset += 1;
                    }
                });
            }
        }
    }
}
