// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::is_aarch64_feature_detected;

use vortex_buffer::BitView;

use crate::filter::Filter;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod scalar;

impl<'a, const NB: usize, T: Copy> Filter<BitView<'a, NB>> for &mut [T] {
    type Output = ();

    fn filter(self, mask: &BitView<'a, NB>) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                // NEON is only faster for sufficiently dense masks.
                match size_of::<T>() {
                    1 | 2 if mask.true_count() < (BitView::<NB>::N / 4) => {
                        // For u8 and u16, the threshold is ~0.25
                        return scalar::filter_scalar(self, mask);
                    }
                    4 if mask.true_count() < (3 * BitView::<NB>::N / 4) => {
                        // For u32, the threshold is ~0.75
                        return scalar::filter_scalar(self, mask);
                    }
                    _ => {}
                }

                return unsafe { neon::filter_neon(self, mask) };
            }
        }

        // Otherwise, fall back to scalar implementation
        scalar::filter_scalar(self, mask);
    }
}
