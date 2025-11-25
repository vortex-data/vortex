// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take function implementations on slices using AVX2 SIMD.

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_dtype::UnsignedPType;

/// Takes the specified indices into a new [`Buffer`] using AVX2 SIMD.
#[allow(dead_code, unused_variables, reason = "TODO(connor): Implement this")]
#[inline]
pub fn take_avx2<T: NativePType, I: UnsignedPType>(buffer: &[T], indices: &[I]) -> Buffer<T> {
    todo!(
        "TODO(connor): Migrate the implementation in \
            vortex-array/src/arrays/primitive/compute/take/avx2.rs"
    )
}
