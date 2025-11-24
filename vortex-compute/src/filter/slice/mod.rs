// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementations of filtering over slices.

pub mod in_place;
pub mod out;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod simd_compress;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use simd_compress::SimdCompress;
