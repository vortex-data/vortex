// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512 multi-limb arithmetic over the split (struct-of-arrays) layout.
//!
//! The trick that the interleaved Arrow layout cannot do: pack 8 values' limbs
//! contiguously and add a whole vector at once, propagating the carry between
//! limbs with mask registers.
//!
//! Carry handling uses the fact that `core::arch::x86_64::__mmask8` is a plain
//! `u8`, so carries OR together with `|` — no `kor` instruction (and thus no
//! AVX-512DQ requirement) needed; everything here is AVX-512F.

use crate::layout::SplitI128;
use crate::layout::SplitI256;
use crate::scalar;

/// i128 add over the split layout, dispatching to AVX-512 when available.
pub fn add_i128(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f confirmed present; all slices share `a.len()`.
            unsafe {
                x86::add_i128_avx512(&a.lo, &a.hi, &b.lo, &b.hi, &mut out.lo, &mut out.hi);
            }
            return;
        }
    }
    scalar::add_i128_soa(a, b, out);
}

/// Unrolled-by-4 i128 add: 32 values per loop iteration. Amortizes loop overhead
/// and exposes 4 independent dependency chains so the core can keep more loads in
/// flight (more memory-level parallelism) - the win in cache/L2/L3 where the
/// single-vector loop is latency-bound, not bandwidth-bound.
pub fn add_i128_u4(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f confirmed present; all slices share `a.len()`.
            unsafe {
                x86::add_i128_avx512_u4(&a.lo, &a.hi, &b.lo, &b.hi, &mut out.lo, &mut out.hi);
            }
            return;
        }
    }
    scalar::add_i128_soa(a, b, out);
}

/// i128 subtract over the split layout, dispatching to AVX-512 when available.
pub fn sub_i128(a: &SplitI128, b: &SplitI128, out: &mut SplitI128) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f confirmed present; all slices share `a.len()`.
            unsafe {
                x86::sub_i128_avx512(&a.lo, &a.hi, &b.lo, &b.hi, &mut out.lo, &mut out.hi);
            }
            return;
        }
    }
    scalar::sub_i128_soa(a, b, out);
}

/// i256 add over the split layout, dispatching to AVX-512 when available.
pub fn add_i256(a: &SplitI256, b: &SplitI256, out: &mut SplitI256) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f confirmed present; all limb slices share `a.len()`.
            unsafe {
                x86::add_i256_avx512(&a.limbs, &b.limbs, &mut out.limbs);
            }
            return;
        }
    }
    scalar::add_i256_soa(a, b, out);
}

/// True if this run will use the AVX-512 path.
pub fn avx512_active() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::arch::is_x86_feature_detected!("avx512f");
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    const LANES: usize = 8;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn add_i128_avx512(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out_lo: &mut [u64],
        out_hi: &mut [u64],
    ) {
        let n = a_lo.len();
        let one = _mm512_set1_epi64(1);
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let alo = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let blo = _mm512_loadu_epi64(b_lo.as_ptr().add(i).cast());
                let slo = _mm512_add_epi64(alo, blo);
                // Unsigned overflow of the low limb => carry into the high limb.
                let carry = _mm512_cmplt_epu64_mask(slo, alo);

                let ahi = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let bhi = _mm512_loadu_epi64(b_hi.as_ptr().add(i).cast());
                let mut shi = _mm512_add_epi64(ahi, bhi);
                shi = _mm512_mask_add_epi64(shi, carry, shi, one);

                _mm512_storeu_epi64(out_lo.as_mut_ptr().add(i).cast(), slo);
                _mm512_storeu_epi64(out_hi.as_mut_ptr().add(i).cast(), shi);
            }
            i += LANES;
        }
        while i < n {
            let (lo, carry) = a_lo[i].overflowing_add(b_lo[i]);
            out_lo[i] = lo;
            out_hi[i] = a_hi[i].wrapping_add(b_hi[i]).wrapping_add(u64::from(carry));
            i += 1;
        }
    }

    /// Unrolled-by-4 variant: 32 values/iteration, 8 independent loads issued
    /// before the dependent carry work, then a single-vector tail.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn add_i128_avx512_u4(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out_lo: &mut [u64],
        out_hi: &mut [u64],
    ) {
        let n = a_lo.len();
        let mut i = 0;
        while i + 4 * LANES <= n {
            unsafe {
                // Issue all 8 low/high loads first so the core has 4 independent
                // chains' worth of outstanding memory requests.
                let mut alo = [_mm512_setzero_si512(); 4];
                let mut blo = [_mm512_setzero_si512(); 4];
                let mut ahi = [_mm512_setzero_si512(); 4];
                let mut bhi = [_mm512_setzero_si512(); 4];
                for j in 0..4 {
                    let o = i + j * LANES;
                    alo[j] = _mm512_loadu_epi64(a_lo.as_ptr().add(o).cast());
                    blo[j] = _mm512_loadu_epi64(b_lo.as_ptr().add(o).cast());
                    ahi[j] = _mm512_loadu_epi64(a_hi.as_ptr().add(o).cast());
                    bhi[j] = _mm512_loadu_epi64(b_hi.as_ptr().add(o).cast());
                }
                for j in 0..4 {
                    let o = i + j * LANES;
                    let slo = _mm512_add_epi64(alo[j], blo[j]);
                    let carry = _mm512_cmplt_epu64_mask(slo, alo[j]);
                    let shi = _mm512_add_epi64(ahi[j], bhi[j]);
                    let shi = _mm512_mask_add_epi64(shi, carry, shi, _mm512_set1_epi64(1));
                    _mm512_storeu_epi64(out_lo.as_mut_ptr().add(o).cast(), slo);
                    _mm512_storeu_epi64(out_hi.as_mut_ptr().add(o).cast(), shi);
                }
            }
            i += 4 * LANES;
        }
        while i + LANES <= n {
            unsafe {
                let alo = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let blo = _mm512_loadu_epi64(b_lo.as_ptr().add(i).cast());
                let slo = _mm512_add_epi64(alo, blo);
                let carry = _mm512_cmplt_epu64_mask(slo, alo);
                let ahi = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let bhi = _mm512_loadu_epi64(b_hi.as_ptr().add(i).cast());
                let shi = _mm512_add_epi64(ahi, bhi);
                let shi = _mm512_mask_add_epi64(shi, carry, shi, _mm512_set1_epi64(1));
                _mm512_storeu_epi64(out_lo.as_mut_ptr().add(i).cast(), slo);
                _mm512_storeu_epi64(out_hi.as_mut_ptr().add(i).cast(), shi);
            }
            i += LANES;
        }
        while i < n {
            let (lo, carry) = a_lo[i].overflowing_add(b_lo[i]);
            out_lo[i] = lo;
            out_hi[i] = a_hi[i].wrapping_add(b_hi[i]).wrapping_add(u64::from(carry));
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn sub_i128_avx512(
        a_lo: &[u64],
        a_hi: &[u64],
        b_lo: &[u64],
        b_hi: &[u64],
        out_lo: &mut [u64],
        out_hi: &mut [u64],
    ) {
        let n = a_lo.len();
        let one = _mm512_set1_epi64(1);
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                let alo = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let blo = _mm512_loadu_epi64(b_lo.as_ptr().add(i).cast());
                let slo = _mm512_sub_epi64(alo, blo);
                // Borrow out of the low limb iff a_lo < b_lo (unsigned).
                let borrow = _mm512_cmplt_epu64_mask(alo, blo);

                let ahi = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let bhi = _mm512_loadu_epi64(b_hi.as_ptr().add(i).cast());
                let mut shi = _mm512_sub_epi64(ahi, bhi);
                shi = _mm512_mask_sub_epi64(shi, borrow, shi, one);

                _mm512_storeu_epi64(out_lo.as_mut_ptr().add(i).cast(), slo);
                _mm512_storeu_epi64(out_hi.as_mut_ptr().add(i).cast(), shi);
            }
            i += LANES;
        }
        while i < n {
            let (lo, borrow) = a_lo[i].overflowing_sub(b_lo[i]);
            out_lo[i] = lo;
            out_hi[i] = a_hi[i]
                .wrapping_sub(b_hi[i])
                .wrapping_sub(u64::from(borrow));
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn add_i256_avx512(a: &[Vec<u64>; 4], b: &[Vec<u64>; 4], out: &mut [Vec<u64>; 4]) {
        let n = a[0].len();
        let one = _mm512_set1_epi64(1);
        let mut i = 0;
        while i + LANES <= n {
            unsafe {
                // limb 0 (no carry in)
                let a0 = _mm512_loadu_epi64(a[0].as_ptr().add(i).cast());
                let b0 = _mm512_loadu_epi64(b[0].as_ptr().add(i).cast());
                let s0 = _mm512_add_epi64(a0, b0);
                let mut carry = _mm512_cmplt_epu64_mask(s0, a0);
                _mm512_storeu_epi64(out[0].as_mut_ptr().add(i).cast(), s0);

                // limbs 1 and 2 (carry in and carry out)
                for k in 1..3 {
                    let ak = _mm512_loadu_epi64(a[k].as_ptr().add(i).cast());
                    let bk = _mm512_loadu_epi64(b[k].as_ptr().add(i).cast());
                    let t = _mm512_add_epi64(ak, bk);
                    let c_ab = _mm512_cmplt_epu64_mask(t, ak);
                    let s = _mm512_mask_add_epi64(t, carry, t, one);
                    let c_carry = _mm512_cmplt_epu64_mask(s, t);
                    carry = c_ab | c_carry;
                    _mm512_storeu_epi64(out[k].as_mut_ptr().add(i).cast(), s);
                }

                // limb 3 (top, carry in only)
                let a3 = _mm512_loadu_epi64(a[3].as_ptr().add(i).cast());
                let b3 = _mm512_loadu_epi64(b[3].as_ptr().add(i).cast());
                let t3 = _mm512_add_epi64(a3, b3);
                let s3 = _mm512_mask_add_epi64(t3, carry, t3, one);
                _mm512_storeu_epi64(out[3].as_mut_ptr().add(i).cast(), s3);
            }
            i += LANES;
        }
        // scalar tail
        while i < n {
            let mut carry = 0u64;
            for k in 0..4 {
                let (s1, c1) = a[k][i].overflowing_add(b[k][i]);
                let (s2, c2) = s1.overflowing_add(carry);
                out[k][i] = s2;
                carry = u64::from(c1 || c2);
            }
            i += 1;
        }
    }
}
