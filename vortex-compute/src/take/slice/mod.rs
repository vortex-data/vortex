// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take function implementations on slices.

use vortex_buffer::Buffer;
use vortex_dtype::UnsignedPType;

use crate::take::Take;

pub mod avx2;
pub mod portable;

/// Specialized implementation for non-nullable indices.
impl<T: Copy, I: UnsignedPType> Take<[I]> for &[T] {
    type Output = Buffer<T>;

    fn take(self, indices: &[I]) -> Buffer<T> {
        // TODO(connor): Make the SIMD implementations bound by `Copy` instead of `NativePType`.
        /*

        #[cfg(vortex_nightly)]
        {
            return portable::take_portable(self, indices);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We just checked that the AVX2 feature in enabled.
                return unsafe { avx2::take_avx2(self, indices) };
            }
        }

        */

        #[allow(unreachable_code, reason = "`vortex_nightly` path returns early")]
        take_scalar(self, indices)
    }
}

#[allow(
    unused,
    reason = "Compiler may see this as unused based on enabled features"
)]
#[inline]
fn take_scalar<T: Copy, I: UnsignedPType>(buffer: &[T], indices: &[I]) -> Buffer<T> {
    indices.iter().map(|idx| buffer[idx.as_()]).collect()
}
