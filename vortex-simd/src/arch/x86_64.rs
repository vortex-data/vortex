//! x86_64 kernels.
//!
//! Two patterns are used here:
//!
//! - **Autovectorize wrappers** for kernels whose generic body LLVM
//!   vectorizes well under `#[target_feature]`. The body lives in
//!   [`crate::kernels::generic`] and each tier wrapper is a one-liner.
//!   The compiler emits SSE2 / AVX2 / AVX-512 code from the same source.
//! - **Hand-tuned intrinsics** for kernels where the autovectorizer cannot
//!   see the right SIMD pattern (mask packing into a bitmap, lane shuffles,
//!   fastlanes-style bit packing). These are the only places the per-arch
//!   source diverges.

// Mask-extraction intrinsics return wider ints whose upper bits are
// guaranteed zero by hardware; truncating the cast is intentional.
#![allow(clippy::cast_possible_truncation)]

use core::arch::x86_64::*;

use crate::kernels::generic;

// ---------- add: autovectorize over one shared body ----------

/// SSE2 `i32` add. Body is shared with every other tier; LLVM emits SSE2
/// here from the generic loop.
///
/// # Safety
/// SSE2 must be available at runtime. Slices must be the same length.
#[target_feature(enable = "sse2")]
pub unsafe fn add_i32_sse2(a: &[i32], b: &[i32], out: &mut [i32]) {
    generic::add_i32(a, b, out)
}

/// AVX2 `i32` add. Same body as SSE2; LLVM emits 256-bit code here.
///
/// # Safety
/// AVX2 must be available at runtime. Slices must be the same length.
#[target_feature(enable = "avx2")]
pub unsafe fn add_i32_avx2(a: &[i32], b: &[i32], out: &mut [i32]) {
    generic::add_i32(a, b, out)
}

/// AVX-512 `i32` add. Same body; LLVM emits 512-bit code here.
///
/// # Safety
/// AVX-512 F must be available at runtime. Slices must be the same length.
#[target_feature(enable = "avx512f")]
pub unsafe fn add_i32_avx512(a: &[i32], b: &[i32], out: &mut [i32]) {
    generic::add_i32(a, b, out)
}

// ---------- eq → bitmap: hand-tuned, autovec can't produce mask packing ----------

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
            // cmpeq_epi32 produces all-ones or all-zeros, so movemask_ps
            // extracts the equality bit from each lane. 4 bits per register;
            // combine into one byte.
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

/// AVX2 `i32` equality → packed bitmap. One full byte per 8 lanes directly
/// from `movemask`.
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

/// AVX-512 `i32` equality → packed bitmap. The mask-register output of
/// `cmpeq` is already in bitmap layout; we just store 2 bytes per chunk.
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
        // SAFETY: idx + 16 <= bulk <= len, out has at least idx/8 + 2 bytes.
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
