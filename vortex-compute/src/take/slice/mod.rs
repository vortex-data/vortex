// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take function implementations on slices.

#![allow(
    unused,
    reason = "Compiler may see things in this module as unused based on enabled features"
)]

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::UnsignedPType;

use crate::take::Take;

pub mod asm_stubs;
pub mod avx2;
pub mod portable;

/// Specialized implementation for non-nullable indices.
impl<T: Copy, I: UnsignedPType> Take<[I]> for &[T] {
    type Output = Buffer<T>;

    fn take(self, indices: &[I]) -> Buffer<T> {
        // Prefer hand-tuned AVX2 on x86 - it's faster than portable SIMD because it uses
        // native 32-bit index gathers (`vpgatherdd`) instead of extending to 64-bit.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We just checked that the AVX2 feature is enabled.
                return unsafe { avx2::take_avx2(self, indices) };
            }
        }

        // Fall back to portable SIMD on non-x86 platforms (ARM, RISC-V) or x86 without AVX2.
        #[cfg(vortex_nightly)]
        {
            return portable::take_portable(self, indices);
        }

        #[allow(unreachable_code, reason = "`vortex_nightly` path returns early")]
        take_scalar(self, indices)
    }
}

pub(crate) fn take_scalar<T: Copy, I: UnsignedPType>(buffer: &[T], indices: &[I]) -> Buffer<T> {
    // NB: The simpler `indices.iter().map(|idx| buff1er[idx.as_()]).collect()` generates suboptimal
    // assembly where the buffer length is repeatedly loaded from the stack on each iteration.

    let mut result = BufferMut::with_capacity(indices.len());
    let ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    // This explicit loop with pointer writes keeps the length in a register and avoids per-element
    // capacity checks from `push()`.
    for (i, idx) in indices.iter().enumerate() {
        // SAFETY: We reserved `indices.len()` capacity, so `ptr.add(i)` is valid.
        unsafe { ptr.add(i).write(buffer[idx.as_()]) };
    }

    // SAFETY: We just wrote exactly `indices.len()` elements.
    unsafe { result.set_len(indices.len()) };
    result.freeze()
}
