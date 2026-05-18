//! x86_64 SIMD kernels for `i32` add and equality.
//!
//! Each kernel processes the bulk of the input in vector chunks and hands
//! the tail off to the scalar fallback so callers never have to worry about
//! alignment or remainder.

// Mask-extraction intrinsics return wider ints whose upper bits are
// guaranteed zero by hardware; truncating the cast is exactly what we want.
#![allow(clippy::cast_possible_truncation)]

use core::arch::x86_64::*;

use crate::kernels::scalar;

// ---------- SSE2 ----------

/// SSE2 `i32` add. SSE2 is the x86_64 baseline, so `#[target_feature]` is
/// mostly a formality, but the unsafe-fn signature documents the contract.
///
/// # Safety
/// SSE2 must be available at runtime. Slices must be the same length.
#[target_feature(enable = "sse2")]
pub unsafe fn add_i32_sse2(a: &[i32], b: &[i32], out: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let bulk = len & !3; // multiple of 4
    let mut idx = 0;
    while idx < bulk {
        // SAFETY: idx + 4 <= bulk <= len, slices are the same length.
        unsafe {
            let vec_a = _mm_loadu_si128(a.as_ptr().add(idx).cast());
            let vec_b = _mm_loadu_si128(b.as_ptr().add(idx).cast());
            let sum = _mm_add_epi32(vec_a, vec_b);
            _mm_storeu_si128(out.as_mut_ptr().add(idx).cast(), sum);
        }
        idx += 4;
    }
    if idx < len {
        scalar::add_i32(&a[idx..], &b[idx..], &mut out[idx..]);
    }
}

/// SSE2 `i32` equality → packed bitmap. `out.len()` must be
/// `(a.len() + 7) / 8`.
///
/// # Safety
/// SSE2 must be available at runtime. Slices and bitmap must be sized
/// according to the contract.
#[target_feature(enable = "sse2")]
pub unsafe fn eq_i32_sse2(a: &[i32], b: &[i32], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len().div_ceil(8));
    let len = a.len();
    out.fill(0);
    let bulk_bytes = len / 8;
    let mut idx = 0;
    for byte_idx in 0..bulk_bytes {
        // SAFETY: idx + 8 <= len.
        unsafe {
            let a_lo = _mm_loadu_si128(a.as_ptr().add(idx).cast());
            let b_lo = _mm_loadu_si128(b.as_ptr().add(idx).cast());
            let a_hi = _mm_loadu_si128(a.as_ptr().add(idx + 4).cast());
            let b_hi = _mm_loadu_si128(b.as_ptr().add(idx + 4).cast());
            let mask_lo = _mm_cmpeq_epi32(a_lo, b_lo);
            let mask_hi = _mm_cmpeq_epi32(a_hi, b_hi);
            // movemask_ps extracts the sign bit of each 32-bit float lane.
            // cmpeq_epi32 produces all-ones or all-zeros, so the sign bit is
            // 1 iff equal. 4 bits per register; combine into one byte.
            let low_nibble = _mm_movemask_ps(_mm_castsi128_ps(mask_lo)) as u32;
            let high_nibble = _mm_movemask_ps(_mm_castsi128_ps(mask_hi)) as u32;
            *out.get_unchecked_mut(byte_idx) = (low_nibble | (high_nibble << 4)) as u8;
        }
        idx += 8;
    }
    while idx < len {
        if a[idx] == b[idx] {
            out[idx / 8] |= 1 << (idx % 8);
        }
        idx += 1;
    }
}

// ---------- AVX2 ----------

/// AVX2 `i32` add, 8 lanes per iteration.
///
/// # Safety
/// AVX2 must be available at runtime. Slices must be the same length.
#[target_feature(enable = "avx2")]
pub unsafe fn add_i32_avx2(a: &[i32], b: &[i32], out: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let bulk = len & !7;
    let mut idx = 0;
    while idx < bulk {
        // SAFETY: idx + 8 <= bulk <= len.
        unsafe {
            let vec_a = _mm256_loadu_si256(a.as_ptr().add(idx).cast());
            let vec_b = _mm256_loadu_si256(b.as_ptr().add(idx).cast());
            let sum = _mm256_add_epi32(vec_a, vec_b);
            _mm256_storeu_si256(out.as_mut_ptr().add(idx).cast(), sum);
        }
        idx += 8;
    }
    if idx < len {
        scalar::add_i32(&a[idx..], &b[idx..], &mut out[idx..]);
    }
}

/// AVX2 `i32` equality → packed bitmap. Produces one full byte per 8 lanes
/// directly from `movemask`.
///
/// # Safety
/// AVX2 must be available at runtime. Slices and bitmap must be sized
/// according to the contract.
#[target_feature(enable = "avx2")]
pub unsafe fn eq_i32_avx2(a: &[i32], b: &[i32], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len().div_ceil(8));
    let len = a.len();
    out.fill(0);
    let bulk_bytes = len / 8;
    let mut idx = 0;
    for byte_idx in 0..bulk_bytes {
        // SAFETY: idx + 8 <= len.
        unsafe {
            let vec_a = _mm256_loadu_si256(a.as_ptr().add(idx).cast());
            let vec_b = _mm256_loadu_si256(b.as_ptr().add(idx).cast());
            let mask = _mm256_cmpeq_epi32(vec_a, vec_b);
            let bits = _mm256_movemask_ps(_mm256_castsi256_ps(mask)) as u32;
            *out.get_unchecked_mut(byte_idx) = bits as u8;
        }
        idx += 8;
    }
    while idx < len {
        if a[idx] == b[idx] {
            out[idx / 8] |= 1 << (idx % 8);
        }
        idx += 1;
    }
}

// ---------- AVX-512 ----------

/// AVX-512 `i32` add, 16 lanes per iteration.
///
/// # Safety
/// AVX-512 F must be available at runtime. Slices must be the same length.
#[target_feature(enable = "avx512f")]
pub unsafe fn add_i32_avx512(a: &[i32], b: &[i32], out: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let bulk = len & !15;
    let mut idx = 0;
    while idx < bulk {
        // SAFETY: idx + 16 <= bulk <= len.
        unsafe {
            let vec_a = _mm512_loadu_si512(a.as_ptr().add(idx).cast());
            let vec_b = _mm512_loadu_si512(b.as_ptr().add(idx).cast());
            let sum = _mm512_add_epi32(vec_a, vec_b);
            _mm512_storeu_si512(out.as_mut_ptr().add(idx).cast(), sum);
        }
        idx += 16;
    }
    if idx < len {
        scalar::add_i32(&a[idx..], &b[idx..], &mut out[idx..]);
    }
}

/// AVX-512 `i32` equality → packed bitmap. The mask register output of
/// `cmpeq` is already in bitmap layout, so we just store 2 bytes per chunk.
///
/// # Safety
/// AVX-512 F must be available at runtime. Slices and bitmap must be sized
/// according to the contract.
#[target_feature(enable = "avx512f")]
pub unsafe fn eq_i32_avx512(a: &[i32], b: &[i32], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len().div_ceil(8));
    let len = a.len();
    out.fill(0);
    let bulk = len & !15;
    let mut idx = 0;
    while idx < bulk {
        // SAFETY: idx + 16 <= bulk <= len, and out has at least idx/8 + 2 bytes.
        unsafe {
            let vec_a = _mm512_loadu_si512(a.as_ptr().add(idx).cast());
            let vec_b = _mm512_loadu_si512(b.as_ptr().add(idx).cast());
            let mask: __mmask16 = _mm512_cmpeq_epi32_mask(vec_a, vec_b);
            let bytes = (mask as u16).to_le_bytes();
            let dst = out.as_mut_ptr().add(idx / 8);
            *dst = bytes[0];
            *dst.add(1) = bytes[1];
        }
        idx += 16;
    }
    while idx < len {
        if a[idx] == b[idx] {
            out[idx / 8] |= 1 << (idx % 8);
        }
        idx += 1;
    }
}
