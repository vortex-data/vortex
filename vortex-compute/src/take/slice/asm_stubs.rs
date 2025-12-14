// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembly inspection stubs for cargo-show-asm.
//!
//! These functions are `#[inline(never)]` wrappers around the take implementations
//! to allow inspecting the generated assembly with cargo-show-asm.
//!
//! # Usage
//!
//! ```bash
//! # Scalar implementations
//! cargo asm -p vortex-compute take_scalar_u32_u32 --rust
//! cargo asm -p vortex-compute take_scalar_u64_u32 --rust
//!
//! # AVX2 implementations
//! cargo asm -p vortex-compute take_avx2_u32_u32 --rust
//! cargo asm -p vortex-compute take_avx2_u64_u32 --rust
//!
//! # Portable SIMD implementations (requires nightly)
//! RUSTFLAGS='--cfg vortex_nightly' cargo +nightly asm -p vortex-compute take_portable_simd_u32_u32 --rust
//! RUSTFLAGS='--cfg vortex_nightly' cargo +nightly asm -p vortex-compute take_portable_simd_u64_u32 --rust
//! ```

#![allow(unused, reason = "These stubs are for assembly inspection only")]

use vortex_buffer::Buffer;

// ============ SCALAR STUBS ============

/// Scalar take: u32 values, u32 indices.
#[inline(never)]
pub fn take_scalar_u32_u32(buffer: &[u32], indices: &[u32]) -> Buffer<u32> {
    super::take_scalar(buffer, indices)
}

/// Scalar take: u64 values, u32 indices.
#[inline(never)]
pub fn take_scalar_u64_u32(buffer: &[u64], indices: &[u32]) -> Buffer<u64> {
    super::take_scalar(buffer, indices)
}

// ============ PORTABLE SIMD STUBS ============

/// Portable SIMD assembly stubs.
#[cfg(vortex_nightly)]
pub mod portable {
    use vortex_buffer::Buffer;

    /// Portable SIMD take: u32 values, u32 indices.
    #[inline(never)]
    pub fn take_portable_simd_u32_u32(buffer: &[u32], indices: &[u32]) -> Buffer<u32> {
        super::super::portable::take_portable_simd::<u32, u32, 16>(buffer, indices)
    }

    /// Portable SIMD take: u64 values, u32 indices.
    #[inline(never)]
    pub fn take_portable_simd_u64_u32(buffer: &[u64], indices: &[u32]) -> Buffer<u64> {
        super::super::portable::take_portable_simd::<u64, u32, 8>(buffer, indices)
    }
}

// ============ AVX2 STUBS ============

/// AVX2 assembly stubs.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2 {
    use vortex_buffer::Buffer;

    /// AVX2 take: u32 values, u32 indices.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available.
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn take_avx2_u32_u32(buffer: &[u32], indices: &[u32]) -> Buffer<u32> {
        unsafe { super::super::avx2::take_avx2(buffer, indices) }
    }

    /// AVX2 take: u64 values, u32 indices.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available.
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn take_avx2_u64_u32(buffer: &[u64], indices: &[u32]) -> Buffer<u64> {
        unsafe { super::super::avx2::take_avx2(buffer, indices) }
    }
}
