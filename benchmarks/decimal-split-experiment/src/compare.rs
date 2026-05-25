// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lexicographic comparison over the split layout, producing an Arrow-compatible
//! bitmap (1 bit per value, LSB-first within each byte).
//!
//! This is the equal-scale fast path: comparing two decimal columns of the same
//! precision/scale is just a signed multi-limb integer comparison. (Differing
//! scales need a rescale first, exactly as Arrow does internally.)
//!
//! The split makes this lane-parallel where Arrow cannot be: an AVX-512 mask
//! compare of 8 high limbs, 8 low limbs, and a couple of mask ops yields a whole
//! output byte at once. The mask register *is* the output byte - lane `k` maps to
//! bit `k`, which matches Arrow's LSB-first bit order, so we store it directly.
//!
//! Only `eq` and `lt` are implemented as primitives; the other four operators are
//! trivial derivations (`gt(a,b)=lt(b,a)`, `ge=!lt`, `le=!gt`, `ne=!eq`) that add
//! no new kernel work.

use crate::layout::SplitI128;
use crate::layout::SplitI256;

/// Number of bytes in a bitmap covering `n` values.
pub fn bitmap_len(n: usize) -> usize {
    n.div_ceil(8)
}

#[inline]
fn set_bit(bitmap: &mut [u8], i: usize, v: bool) {
    if v {
        bitmap[i / 8] |= 1 << (i % 8);
    }
}

/// Read a bit back out (for tests / verification).
pub fn get_bit(bitmap: &[u8], i: usize) -> bool {
    (bitmap[i / 8] >> (i % 8)) & 1 == 1
}

// ---- scalar references -------------------------------------------------------

pub fn lt_i128_scalar(a: &SplitI128, b: &SplitI128, out: &mut [u8]) {
    out.iter_mut().for_each(|byte| *byte = 0);
    for i in 0..a.len() {
        let ah = a.hi[i] as i64;
        let bh = b.hi[i] as i64;
        let lt = ah < bh || (ah == bh && a.lo[i] < b.lo[i]);
        set_bit(out, i, lt);
    }
}

pub fn eq_i128_scalar(a: &SplitI128, b: &SplitI128, out: &mut [u8]) {
    out.iter_mut().for_each(|byte| *byte = 0);
    for i in 0..a.len() {
        set_bit(out, i, a.hi[i] == b.hi[i] && a.lo[i] == b.lo[i]);
    }
}

pub fn lt_i256_scalar(a: &SplitI256, b: &SplitI256, out: &mut [u8]) {
    out.iter_mut().for_each(|byte| *byte = 0);
    for i in 0..a.len() {
        let lt = lexicographic_lt_i256(a, b, i);
        set_bit(out, i, lt);
    }
}

fn lexicographic_lt_i256(a: &SplitI256, b: &SplitI256, i: usize) -> bool {
    // Top limb signed, the rest unsigned, most significant first.
    let a3 = a.limbs[3][i] as i64;
    let b3 = b.limbs[3][i] as i64;
    if a3 != b3 {
        return a3 < b3;
    }
    for k in (0..3).rev() {
        let ak = a.limbs[k][i];
        let bk = b.limbs[k][i];
        if ak != bk {
            return ak < bk;
        }
    }
    false
}

// ---- constant-limb-aware fast paths ------------------------------------------

/// Whether a limb stream is constant, returning the value if so. In the real
/// system this is recorded by the compression encoding, so it is known for free;
/// this scan exists only to drive tests/benchmarks.
pub fn const_value(s: &[u64]) -> Option<u64> {
    s.first()
        .copied()
        .filter(|&first| s.iter().all(|&v| v == first))
}

/// `lt` when both columns have a *known constant* high limb (from the encoding).
///
/// - If the two high constants differ, every result is identical: the output is
///   a constant bitmap filled in O(1) - no per-element work at all.
/// - If they are equal, the high limbs cancel and we compare only the low limbs
///   (unsigned), never touching the high streams.
///
/// Arrow cannot collapse either case: it must scan all values regardless.
pub fn lt_i128_const_hi(
    a_lo: &[u64],
    a_hi_const: u64,
    b_lo: &[u64],
    b_hi_const: u64,
    out: &mut [u8],
) {
    let n = a_lo.len();
    let ah = a_hi_const as i64;
    let bh = b_hi_const as i64;
    if ah != bh {
        // Whole-column constant result.
        let fill = if ah < bh { 0xFFu8 } else { 0x00u8 };
        for byte in out.iter_mut() {
            *byte = fill;
        }
        // Clear bits past the last value in the final byte.
        if fill != 0 && n % 8 != 0 {
            out[n / 8] = (1u8 << (n % 8)) - 1;
        }
        return;
    }
    // High limbs equal: pure unsigned low-limb comparison.
    lt_u64_unsigned(a_lo, b_lo, out);
}

/// Unsigned `lt` over two u64 streams into a bitmap (AVX-512 when available).
pub fn lt_u64_unsigned(a: &[u64], b: &[u64], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present; slices share length.
            unsafe {
                x86::lt_u64_avx512(a, b, out);
            }
            return;
        }
    }
    out.iter_mut().for_each(|byte| *byte = 0);
    for i in 0..a.len() {
        set_bit(out, i, a[i] < b[i]);
    }
}

/// Block-wise `lt` exploiting *partial* constancy. `block` must be a multiple of
/// 8 so block boundaries are byte-aligned in the output bitmap. Per block: if
/// both columns' high limbs are constant there, use the constant fast path
/// (skip the high stream / fill); otherwise fall back to the full lexicographic
/// kernel for that block.
pub fn lt_i128_blockwise(
    a: &SplitI128,
    a_hi_const: &[Option<u64>],
    b: &SplitI128,
    b_hi_const: &[Option<u64>],
    block: usize,
    out: &mut [u8],
) {
    debug_assert_eq!(block % 8, 0, "block must be byte-aligned");
    let n = a.len();
    for (k, (&ca, &cb)) in a_hi_const.iter().zip(b_hi_const).enumerate() {
        let start = k * block;
        let end = (start + block).min(n);
        if start >= end {
            break;
        }
        let byte_lo = start / 8;
        let byte_hi = end.div_ceil(8);
        let out_block = &mut out[byte_lo..byte_hi];
        match (ca, cb) {
            (Some(ca), Some(cb)) => {
                lt_i128_const_hi(&a.lo[start..end], ca, &b.lo[start..end], cb, out_block);
            }
            // Varying block: full lexicographic kernel, written straight into
            // the block's bitmap bytes (no allocation, no copy).
            _ => lt_i128_slices(
                &a.lo[start..end],
                &a.hi[start..end],
                &b.lo[start..end],
                &b.hi[start..end],
                out_block,
            ),
        }
    }
}

// ---- dispatch ----------------------------------------------------------------

pub fn lt_i128(a: &SplitI128, b: &SplitI128, out: &mut [u8]) {
    lt_i128_slices(&a.lo, &a.hi, &b.lo, &b.hi, out);
}

/// Slice-based `lt` core so block-wise code can write into a sub-bitmap with no
/// allocation.
pub fn lt_i128_slices(a_lo: &[u64], a_hi: &[u64], b_lo: &[u64], b_hi: &[u64], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present; slices share length, out is bitmap-sized.
            unsafe {
                x86::lt_i128_avx512(a_lo, a_hi, b_lo, b_hi, out);
            }
            return;
        }
    }
    out.iter_mut().for_each(|byte| *byte = 0);
    for i in 0..a_lo.len() {
        let ah = a_hi[i] as i64;
        let bh = b_hi[i] as i64;
        set_bit(out, i, ah < bh || (ah == bh && a_lo[i] < b_lo[i]));
    }
}

pub fn eq_i128(a: &SplitI128, b: &SplitI128, out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present; slices share length, out is bitmap-sized.
            unsafe {
                x86::eq_i128_avx512(&a.lo, &a.hi, &b.lo, &b.hi, out);
            }
            return;
        }
    }
    eq_i128_scalar(a, b, out);
}

pub fn lt_i256(a: &SplitI256, b: &SplitI256, out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present; all limb slices share length.
            unsafe {
                x86::lt_i256_avx512(&a.limbs, &b.limbs, out);
            }
            return;
        }
    }
    lt_i256_scalar(a, b, out);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    const LANES: usize = 8;

    /// Unsigned `lt` over two u64 streams - the kernel used once the constant
    /// high limbs have been established equal, so only low limbs remain.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn lt_u64_avx512(a: &[u64], b: &[u64], out: &mut [u8]) {
        let n = a.len();
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let av = _mm512_loadu_epi64(a.as_ptr().add(i).cast());
                let bv = _mm512_loadu_epi64(b.as_ptr().add(i).cast());
                out[i / 8] = _mm512_cmplt_epu64_mask(av, bv);
            }
            i += LANES;
        }
        for j in i..n {
            if a[j] < b[j] {
                out[j / 8] |= 1 << (j % 8);
            } else {
                out[j / 8] &= !(1 << (j % 8));
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn lt_i128_avx512(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out: &mut [u8],
    ) {
        let n = a_lo.len();
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let ah = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let bh = _mm512_loadu_epi64(b_hi.as_ptr().add(i).cast());
                let al = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let bl = _mm512_loadu_epi64(b_lo.as_ptr().add(i).cast());

                let hi_lt = _mm512_cmplt_epi64_mask(ah, bh); // signed
                let hi_eq = _mm512_cmpeq_epi64_mask(ah, bh);
                let lo_lt = _mm512_cmplt_epu64_mask(al, bl); // unsigned
                let mask = hi_lt | (hi_eq & lo_lt);
                out[i / 8] = mask;
            }
            i += LANES;
        }
        scalar_tail_lt_i128(a_lo, a_hi, b_lo, b_hi, out, i, n);
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn eq_i128_avx512(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out: &mut [u8],
    ) {
        let n = a_lo.len();
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let ah = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let bh = _mm512_loadu_epi64(b_hi.as_ptr().add(i).cast());
                let al = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let bl = _mm512_loadu_epi64(b_lo.as_ptr().add(i).cast());
                let mask = _mm512_cmpeq_epi64_mask(ah, bh) & _mm512_cmpeq_epi64_mask(al, bl);
                out[i / 8] = mask;
            }
            i += LANES;
        }
        for j in i..n {
            if a_hi[j] == b_hi[j] && a_lo[j] == b_lo[j] {
                out[j / 8] |= 1 << (j % 8);
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn lt_i256_avx512(a: &[Vec<u64>; 4], b: &[Vec<u64>; 4], out: &mut [u8]) {
        let n = a[0].len();
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let a3 = _mm512_loadu_epi64(a[3].as_ptr().add(i).cast());
                let b3 = _mm512_loadu_epi64(b[3].as_ptr().add(i).cast());
                let a2 = _mm512_loadu_epi64(a[2].as_ptr().add(i).cast());
                let b2 = _mm512_loadu_epi64(b[2].as_ptr().add(i).cast());
                let a1 = _mm512_loadu_epi64(a[1].as_ptr().add(i).cast());
                let b1 = _mm512_loadu_epi64(b[1].as_ptr().add(i).cast());
                let a0 = _mm512_loadu_epi64(a[0].as_ptr().add(i).cast());
                let b0 = _mm512_loadu_epi64(b[0].as_ptr().add(i).cast());

                // Most significant limb is signed; the rest unsigned.
                let l3_lt = _mm512_cmplt_epi64_mask(a3, b3);
                let l3_eq = _mm512_cmpeq_epi64_mask(a3, b3);
                let l2_lt = _mm512_cmplt_epu64_mask(a2, b2);
                let l2_eq = _mm512_cmpeq_epi64_mask(a2, b2);
                let l1_lt = _mm512_cmplt_epu64_mask(a1, b1);
                let l1_eq = _mm512_cmpeq_epi64_mask(a1, b1);
                let l0_lt = _mm512_cmplt_epu64_mask(a0, b0);

                let mask = l3_lt | (l3_eq & (l2_lt | (l2_eq & (l1_lt | (l1_eq & l0_lt)))));
                out[i / 8] = mask;
            }
            i += LANES;
        }
        for j in i..n {
            if super::lexicographic_lt_i256_at(a, b, j) {
                out[j / 8] |= 1 << (j % 8);
            }
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn scalar_tail_lt_i128(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out: &mut [u8],
        start: usize,
        n: usize,
    ) {
        for j in start..n {
            let ah = a_hi[j] as i64;
            let bh = b_hi[j] as i64;
            if ah < bh || (ah == bh && a_lo[j] < b_lo[j]) {
                out[j / 8] |= 1 << (j % 8);
            }
        }
    }
}

/// Tail helper shared with the SIMD path (free function so the x86 module can
/// reach it without duplicating the lexicographic logic).
#[cfg(target_arch = "x86_64")]
fn lexicographic_lt_i256_at(a: &[Vec<u64>; 4], b: &[Vec<u64>; 4], i: usize) -> bool {
    let a3 = a[3][i] as i64;
    let b3 = b[3][i] as i64;
    if a3 != b3 {
        return a3 < b3;
    }
    for k in (0..3).rev() {
        if a[k][i] != b[k][i] {
            return a[k][i] < b[k][i];
        }
    }
    false
}
