// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use vortex_buffer::BitView;
use vortex_mask::Mask;

use crate::filter::Filter;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod scalar;

impl<'a, const NB: usize, T: Copy> Filter<BitView<'a, NB>> for &mut [T] {
    type Output = Self;

    fn filter(self, mask: &BitView<'a, NB>) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // NEON is only faster for sufficiently dense masks.
                match size_of::<T>() {
                    1 | 2 if mask.true_count() < (BitView::<NB>::N / 4) => {
                        // For u8 and u16, the threshold is ~0.25
                        scalar::filter_scalar(self, mask);
                        return &mut self[..mask.true_count()];
                    }
                    4 if mask.true_count() < (3 * BitView::<NB>::N / 4) => {
                        // For u32, the threshold is ~0.75
                        scalar::filter_scalar(self, mask);
                        return &mut self[..mask.true_count()];
                    }
                    _ => {}
                }

                unsafe { neon::filter_neon(self, mask) }
                return &mut self[..mask.true_count()];
            }
        }

        // Otherwise, fall back to scalar implementation
        scalar::filter_scalar(self, mask);
        &mut self[..mask.true_count()]
    }
}

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
