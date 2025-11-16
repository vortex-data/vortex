// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::filter::Filter;
use std::arch::is_aarch64_feature_detected;
use vortex_buffer::BitView;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod scalar;

impl<'a, const NB: usize, T: Copy> Filter<BitView<'a, NB>> for &mut [T] {
    type Output = ();

    fn filter(self, mask: &BitView<'a, NB>) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                return unsafe { neon::filter_neon(self, mask) };
            }
        }

        // Otherwise, fall back to scalar implementation
        scalar::filter_scalar(self, mask);
    }
}
