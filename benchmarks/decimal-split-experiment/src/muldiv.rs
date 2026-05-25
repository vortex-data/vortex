// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal multiply and divide.
//!
//! **Multiply** produces the low 128 bits of the product (wrapping), which is
//! exactly `i128::wrapping_mul` and Arrow's `mul_wrapping`. The decimal scale of
//! the result is `scale_a + scale_b`, but that is metadata - the integer kernel
//! is the same regardless. Unlike add/compare (memory-bandwidth bound), multiply
//! is compute-bound, so an AVX-512 kernel that does 8 products at once with
//! `vpmullq` + a 32-bit `mulhi` decomposition can actually pull ahead of Arrow's
//! per-element scalar multiply - the split layout is what lets us feed the
//! limbs to those lane-parallel multiplies.
//!
//! **Divide** is 128-bit integer division. There is no SIMD divide primitive and
//! the hi/lo split gives no leverage: both "modes" reduce to the same scalar
//! divide. It is benchmarked anyway for an honest, complete picture. With scale
//! 0 (used by the benchmarks) Arrow's decimal `div` is plain integer division.

use crate::layout::SplitI128;

// ---- multiply: low-128-bit product -------------------------------------------

/// Interleaved (array-of-structs) multiply - the layout Arrow uses.
pub fn mul_i128_aos(a: &[i128], b: &[i128], out: &mut [i128]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_mul(b[i]);
    }
}

/// Split (struct-of-arrays) multiply, scalar, via the limb product formula.
pub fn mul_i128_soa_scalar(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    for i in 0..a.len() {
        let (lo, hi) = mul_lo128(a.lo[i], a.hi[i], b.lo[i], b.hi[i]);
        out.lo[i] = lo;
        out.hi[i] = hi;
    }
}

/// Low 128 bits of the product of two i128 bit-patterns, as `(lo, hi)` limbs.
///
/// `a*b mod 2^128 = a_lo*b_lo + ((a_lo*b_hi + a_hi*b_lo) << 64)`; the
/// `a_hi*b_hi` term lands entirely above bit 128 and is dropped. Two's-complement
/// low bits equal the unsigned low bits, so unsigned limb math is correct.
#[inline]
fn mul_lo128(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64) -> (u64, u64) {
    let ll = u128::from(a_lo) * u128::from(b_lo);
    let cross = a_lo
        .wrapping_mul(b_hi)
        .wrapping_add(a_hi.wrapping_mul(b_lo));
    let res = ll.wrapping_add(u128::from(cross) << 64);
    (res as u64, (res >> 64) as u64)
}

/// Split multiply dispatching to AVX-512 (needs `avx512f` + `avx512dq` for the
/// 64-bit `vpmullq`).
pub fn mul_i128(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq")
        {
            // SAFETY: avx512f+avx512dq present; slices share length.
            unsafe {
                x86::mul_i128_avx512(&a.lo, &a.hi, &b.lo, &b.hi, &mut out.lo, &mut out.hi);
            }
            return;
        }
    }
    mul_i128_soa_scalar(a, b, out);
}

// ---- divide ------------------------------------------------------------------

/// Interleaved 128-bit integer division (guards divide-by-zero by emitting 0).
pub fn div_i128_aos(a: &[i128], b: &[i128], out: &mut [i128]) {
    for i in 0..a.len() {
        out[i] = if b[i] == 0 {
            0
        } else {
            a[i].wrapping_div(b[i])
        };
    }
}

/// Split-layout division. There is no SIMD divide and the split does not help,
/// so this reconstructs and divides - identical work to the AoS path, included
/// to show the two "modes" converge for division.
pub fn div_i128_soa(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    for i in 0..a.len() {
        let x = reassemble(a.lo[i], a.hi[i]);
        let y = reassemble(b.lo[i], b.hi[i]);
        let q = if y == 0 { 0 } else { x.wrapping_div(y) };
        out.lo[i] = q as u128 as u64;
        out.hi[i] = ((q as u128) >> 64) as u64;
    }
}

#[inline]
fn reassemble(lo: u64, hi: u64) -> i128 {
    (((hi as u128) << 64) | (lo as u128)) as i128
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    const LANES: usize = 8;

    /// 8-wide low-128 product over split limbs.
    ///
    /// `out_lo = lo64(a_lo*b_lo)`,
    /// `out_hi = hi64(a_lo*b_lo) + lo64(a_lo*b_hi) + lo64(a_hi*b_lo)`.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn mul_i128_avx512(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out_lo: &mut [u64],
        out_hi: &mut [u64],
    ) {
        let n = a_lo.len();
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let alo = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let ahi = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let blo = _mm512_loadu_epi64(b_lo.as_ptr().add(i).cast());
                let bhi = _mm512_loadu_epi64(b_hi.as_ptr().add(i).cast());

                let low = _mm512_mullo_epi64(alo, blo); // bits 0..63
                let hi_ll = mulhi_u64(alo, blo);
                let cross =
                    _mm512_add_epi64(_mm512_mullo_epi64(alo, bhi), _mm512_mullo_epi64(ahi, blo));
                let high = _mm512_add_epi64(hi_ll, cross); // bits 64..127

                _mm512_storeu_epi64(out_lo.as_mut_ptr().add(i).cast(), low);
                _mm512_storeu_epi64(out_hi.as_mut_ptr().add(i).cast(), high);
            }
            i += LANES;
        }
        for j in i..n {
            let (lo, hi) = super::mul_lo128(a_lo[j], a_hi[j], b_lo[j], b_hi[j]);
            out_lo[j] = lo;
            out_hi[j] = hi;
        }
    }

    /// High 64 bits of the unsigned product of two u64 lanes, via four 32x32
    /// `vpmuludq` products and the standard carry recombination.
    #[target_feature(enable = "avx512f")]
    fn mulhi_u64(x: __m512i, y: __m512i) -> __m512i {
        let xh = _mm512_srli_epi64(x, 32);
        let yh = _mm512_srli_epi64(y, 32);
        // vpmuludq multiplies the low 32 bits of each lane.
        let t0 = _mm512_mul_epu32(x, y); // xl*yl
        let t1 = _mm512_mul_epu32(xh, y); // xh*yl
        let t2 = _mm512_mul_epu32(x, yh); // xl*yh
        let t3 = _mm512_mul_epu32(xh, yh); // xh*yh

        let mask32 = _mm512_set1_epi64(0xFFFF_FFFF);
        let mid = _mm512_add_epi64(
            _mm512_add_epi64(_mm512_srli_epi64(t0, 32), _mm512_and_si512(t1, mask32)),
            _mm512_and_si512(t2, mask32),
        );
        _mm512_add_epi64(
            _mm512_add_epi64(t3, _mm512_srli_epi64(t1, 32)),
            _mm512_add_epi64(_mm512_srli_epi64(t2, 32), _mm512_srli_epi64(mid, 32)),
        )
    }
}
