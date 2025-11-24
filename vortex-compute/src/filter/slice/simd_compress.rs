// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic SIMD compress trait for numeric types.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Trait for types that can use SIMD `VCOMPRESSD` operations.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub trait SimdCompress: Copy {
    /// Width of this type in bits.
    const WIDTH: usize;

    /// Number of elements that fit in a 512-bit AVX-512 register.
    const ELEMENTS_PER_VECTOR: usize = 512 / Self::WIDTH;

    /// Number of bytes to read from the mask array for a full vector.
    const MASK_BYTES: usize = Self::ELEMENTS_PER_VECTOR / 8;

    /// The mask type used for this element size (u8, u16, u32, or u64).
    type MaskType: Copy;

    /// Type-specific compress operation.
    ///
    /// # Safety
    ///
    /// This function requires the appropriate SIMD instruction set to be available at runtime.
    ///
    /// - For AVX-512F types (32 and 64 bit), the CPU must support AVX-512F.
    /// - For AVX-512VBMI2 types (8 and 16 bit), the CPU must support AVX-512VBMI2.
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i;

    /// Read mask from byte array at given byte offset.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `mask_ptr` points to valid memory with at least `byte_offset + Self::MASK_BYTES` readable
    ///   bytes.
    /// - The pointer arithmetic `mask_ptr.add(byte_offset)` does not overflow.
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType;

    /// Count the number of set bits in a mask.
    fn count_ones(mask: Self::MaskType) -> usize;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 32-bit types (16 elements per vector, AVX-512F)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for i32 {
    const WIDTH: usize = 32;
    type MaskType = u16;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi32(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u16) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for u32 {
    const WIDTH: usize = 32;
    type MaskType = u16;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi32(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u16) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for f32 {
    const WIDTH: usize = 32;
    type MaskType = u16;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        unsafe {
            let float_vec = std::mem::transmute::<__m512i, __m512>(vec);
            let compressed = _mm512_maskz_compress_ps(mask, float_vec);
            std::mem::transmute::<__m512, __m512i>(compressed)
        }
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u16) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64-bit types (8 elements per vector, AVX-512F)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for i64 {
    const WIDTH: usize = 64;
    type MaskType = u8;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi64(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { *mask_ptr.add(byte_offset) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for u64 {
    const WIDTH: usize = 64;
    type MaskType = u8;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi64(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { *mask_ptr.add(byte_offset) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for f64 {
    const WIDTH: usize = 64;
    type MaskType = u8;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        unsafe {
            let double_vec = std::mem::transmute::<__m512i, __m512d>(vec);
            let compressed = _mm512_maskz_compress_pd(mask, double_vec);
            std::mem::transmute::<__m512d, __m512i>(compressed)
        }
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { *mask_ptr.add(byte_offset) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 16-bit types (32 elements per vector, AVX-512VBMI2)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for i16 {
    const WIDTH: usize = 16;
    type MaskType = u32;

    #[target_feature(enable = "avx512vbmi2")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi16(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u32) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for u16 {
    const WIDTH: usize = 16;
    type MaskType = u32;

    #[target_feature(enable = "avx512vbmi2")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi16(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u32) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 8-bit types (64 elements per vector, AVX-512VBMI2)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for i8 {
    const WIDTH: usize = 8;
    type MaskType = u64;

    #[target_feature(enable = "avx512vbmi2")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi8(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u64) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdCompress for u8 {
    const WIDTH: usize = 8;
    type MaskType = u64;

    #[target_feature(enable = "avx512vbmi2")]
    #[inline]
    unsafe fn compress_vector(mask: Self::MaskType, vec: __m512i) -> __m512i {
        _mm512_maskz_compress_epi8(mask, vec)
    }

    #[inline]
    unsafe fn read_mask(mask_ptr: *const u8, byte_offset: usize) -> Self::MaskType {
        unsafe { core::ptr::read_unaligned(mask_ptr.add(byte_offset) as *const u64) }
    }

    #[inline]
    fn count_ones(mask: Self::MaskType) -> usize {
        mask.count_ones() as usize
    }
}
