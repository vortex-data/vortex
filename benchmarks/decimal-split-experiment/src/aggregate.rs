// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reductions over the split layout: sum (with overflow-safe widening) and
//! min/max.
//!
//! Overflow is the whole story for sum. Arrow's `sum` over a Decimal128 column
//! accumulates back into i128 and **wraps silently** on overflow. Vortex instead
//! widens: it accumulates into i256 and only then checks the result against the
//! output precision. We reproduce the widening accumulator here.
//!
//! The split makes the widening cheap and exact. A two's-complement i128 value
//! decomposes as `v = (hi as i64) * 2^64 + (lo as u64)`, so
//! `Σ v = (Σ hi_i) * 2^64 + (Σ lo_i)` where the low sum is unsigned and the high
//! sum is signed. Accumulating the two limb streams into 128-bit partials and
//! combining gives an exact i256 total with no per-element 256-bit math. For
//! `n < 2^64` values it cannot overflow i256, so the widened sum is always exact.

use arrow_buffer::i256;

use crate::layout::SplitI128;

/// Exact widening sum of an i128 column into i256 (never overflows for
/// realistic `n`). This is the overflow-safe semantics Vortex guarantees.
pub fn sum_i128_widening_scalar(a: &SplitI128) -> i256 {
    let mut s_lo: u128 = 0; // Σ of low (unsigned) limbs
    let mut s_hi: i128 = 0; // Σ of high (signed) limbs
    for i in 0..a.len() {
        s_lo = s_lo.wrapping_add(u128::from(a.lo[i]));
        s_hi = s_hi.wrapping_add(i128::from(a.hi[i] as i64));
    }
    combine(s_hi, s_lo)
}

/// Naive same-width sum (wrapping i128), shown only to demonstrate that the
/// non-widening accumulator silently overflows - the bug the widening avoids.
pub fn sum_i128_naive_wrapping(a: &SplitI128) -> i128 {
    let mut acc: i128 = 0;
    for v in a.to_aos() {
        acc = acc.wrapping_add(v);
    }
    acc
}

/// Exact widening sum, dispatching to the AVX-512 lane-accumulator when present.
pub fn sum_i128_widening(a: &SplitI128) -> i256 {
    sum_widening_slices(&a.lo, &a.hi)
}

/// Slice-based exact widening sum (so block-wise code can sum sub-ranges).
pub fn sum_widening_slices(lo: &[u64], hi: &[u64]) -> i256 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present; lo/hi share length.
            return unsafe { x86_sum::sum_widening_avx512(lo, hi) };
        }
    }
    let mut s_lo: u128 = 0;
    let mut s_hi: i128 = 0;
    for i in 0..lo.len() {
        s_lo = s_lo.wrapping_add(u128::from(lo[i]));
        s_hi = s_hi.wrapping_add(i128::from(hi[i] as i64));
    }
    combine(s_hi, s_lo)
}

/// Block-wise sum exploiting *partial* constancy. Real columnar engines carry
/// per-chunk stats; `hi_const_per_block[k]` is `Some(c)` when the high limb is
/// constant `c` within block `k` (so that block skips the high stream entirely)
/// and `None` when it varies (full widening for that block). The column as a
/// whole need not be constant - this is the realistic case.
pub fn sum_i128_blockwise(
    lo: &[u64],
    hi: &[u64],
    hi_const_per_block: &[Option<u64>],
    block: usize,
) -> i256 {
    let n = lo.len();
    let mut total = i256::ZERO;
    for (k, &meta) in hi_const_per_block.iter().enumerate() {
        let start = k * block;
        let end = (start + block).min(n);
        if start >= end {
            break;
        }
        let part = match meta {
            // Constant high block: read only the low limbs.
            Some(c) => sum_i128_const_hi(&lo[start..end], c),
            // Varying block: full widening over both limbs.
            None => sum_widening_slices(&lo[start..end], &hi[start..end]),
        };
        total = total.wrapping_add(part);
    }
    total
}

/// Fast path for the small-decimal case where the high limb is a known constant
/// zero: only the low limbs are summed, halving memory traffic. Caller must have
/// established `hi == 0` (e.g. from stats / the split's hi-limb bit-width).
pub fn sum_i128_lo_only(a: &SplitI128) -> i256 {
    u128_to_i256(sum_lo_u128(&a.lo))
}

/// Sum when the high limb is a *known constant* `hi_const` (recorded by the
/// compression encoding, so known for free - no scan). Reads only the low
/// limbs; the constant high contribution is folded analytically as
/// `hi_const * N * 2^64`. The entire high stream is skipped.
///
/// Arrow cannot do this: its high bytes are interleaved with the low bytes, so
/// it must read every value in full regardless of how trivial the high half is.
pub fn sum_i128_const_hi(lo: &[u64], hi_const: u64) -> i256 {
    let s_lo = sum_lo_u128(lo);
    let n = lo.len() as i128;
    let s_hi = i128::from(hi_const as i64).wrapping_mul(n);
    combine(s_hi, s_lo)
}

/// Exact unsigned sum of a low-limb stream (AVX-512 lane+carry when available).
fn sum_lo_u128(lo: &[u64]) -> u128 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present.
            return unsafe { x86_sum::sum_u64_lane_carry_avx512(lo) };
        }
    }
    lo.iter()
        .fold(0u128, |acc, &v| acc.wrapping_add(u128::from(v)))
}

/// `Σhi * 2^64 + Σlo` as an exact i256.
fn combine(s_hi: i128, s_lo: u128) -> i256 {
    let hi = i256::from_i128(s_hi);
    let two64 = i256::from_i128(1i128 << 64);
    let lo = u128_to_i256(s_lo);
    hi.wrapping_mul(two64).wrapping_add(lo)
}

fn u128_to_i256(v: u128) -> i256 {
    // i256::from_i128 only covers signed 128-bit; split the unsigned value so the
    // top bit is never misread as a sign.
    let low = (v & u128::from(u64::MAX)) as u64;
    let high = (v >> 64) as u64;
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&low.to_le_bytes());
    bytes[8..16].copy_from_slice(&high.to_le_bytes());
    i256::from_le_bytes(bytes)
}

// ---- min / max ---------------------------------------------------------------

/// Minimum of an i128 column, scalar over the split layout.
pub fn min_i128_scalar(a: &SplitI128) -> Option<i128> {
    if a.is_empty() {
        return None;
    }
    let mut best_hi = a.hi[0];
    let mut best_lo = a.lo[0];
    for i in 1..a.len() {
        if lt(a.hi[i], a.lo[i], best_hi, best_lo) {
            best_hi = a.hi[i];
            best_lo = a.lo[i];
        }
    }
    Some(reassemble(best_hi, best_lo))
}

/// Maximum of an i128 column, scalar over the split layout.
pub fn max_i128_scalar(a: &SplitI128) -> Option<i128> {
    if a.is_empty() {
        return None;
    }
    let mut best_hi = a.hi[0];
    let mut best_lo = a.lo[0];
    for i in 1..a.len() {
        if lt(best_hi, best_lo, a.hi[i], a.lo[i]) {
            best_hi = a.hi[i];
            best_lo = a.lo[i];
        }
    }
    Some(reassemble(best_hi, best_lo))
}

#[inline]
fn lt(a_hi: u64, a_lo: u64, b_hi: u64, b_lo: u64) -> bool {
    let ah = a_hi as i64;
    let bh = b_hi as i64;
    ah < bh || (ah == bh && a_lo < b_lo)
}

#[inline]
fn reassemble(hi: u64, lo: u64) -> i128 {
    (((hi as u128) << 64) | (lo as u128)) as i128
}

/// Minimum dispatching to AVX-512 lane-parallel reduction when available.
pub fn min_i128(a: &SplitI128) -> Option<i128> {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 16 && std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present, len >= 16 so the 8-lane init is in bounds.
            return Some(unsafe { x86::minmax_i128_avx512(&a.lo, &a.hi, true) });
        }
    }
    min_i128_scalar(a)
}

/// Maximum dispatching to AVX-512 lane-parallel reduction when available.
pub fn max_i128(a: &SplitI128) -> Option<i128> {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 16 && std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f present, len >= 16 so the 8-lane init is in bounds.
            return Some(unsafe { x86::minmax_i128_avx512(&a.lo, &a.hi, false) });
        }
    }
    max_i128_scalar(a)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    use super::lt;
    use super::reassemble;

    const LANES: usize = 8;

    /// Lane-parallel min/max: keep 8 running candidates, blend in lane-parallel
    /// using the lexicographic compare mask, then reduce the 8 lanes and the tail
    /// in scalar. `want_min` selects min vs max.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn minmax_i128_avx512(a_lo: &[u64], a_hi: &[u64], want_min: bool) -> i128 {
        let n = a_lo.len();
        unsafe {
            let mut best_lo = _mm512_loadu_epi64(a_lo.as_ptr().cast());
            let mut best_hi = _mm512_loadu_epi64(a_hi.as_ptr().cast());
            let mut i = LANES;
            while i + LANES <= n {
                let lo = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let hi = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());

                // mask = lanes where `incoming` should replace `best`.
                let hi_eq = _mm512_cmpeq_epi64_mask(hi, best_hi);
                let take = if want_min {
                    let hi_lt = _mm512_cmplt_epi64_mask(hi, best_hi);
                    let lo_lt = _mm512_cmplt_epu64_mask(lo, best_lo);
                    hi_lt | (hi_eq & lo_lt)
                } else {
                    let hi_gt = _mm512_cmpgt_epi64_mask(hi, best_hi);
                    let lo_gt = _mm512_cmpgt_epu64_mask(lo, best_lo);
                    hi_gt | (hi_eq & lo_gt)
                };
                best_lo = _mm512_mask_blend_epi64(take, best_lo, lo);
                best_hi = _mm512_mask_blend_epi64(take, best_hi, hi);
                i += LANES;
            }

            // Spill the 8 candidates and reduce in scalar, then fold the tail.
            let mut lanes_lo = [0u64; LANES];
            let mut lanes_hi = [0u64; LANES];
            _mm512_storeu_epi64(lanes_lo.as_mut_ptr().cast(), best_lo);
            _mm512_storeu_epi64(lanes_hi.as_mut_ptr().cast(), best_hi);

            let mut r_hi = lanes_hi[0];
            let mut r_lo = lanes_lo[0];
            for k in 1..LANES {
                if replace(want_min, lanes_hi[k], lanes_lo[k], r_hi, r_lo) {
                    r_hi = lanes_hi[k];
                    r_lo = lanes_lo[k];
                }
            }
            for j in i..n {
                if replace(want_min, a_hi[j], a_lo[j], r_hi, r_lo) {
                    r_hi = a_hi[j];
                    r_lo = a_lo[j];
                }
            }
            reassemble(r_hi, r_lo)
        }
    }

    #[inline]
    fn replace(want_min: bool, cand_hi: u64, cand_lo: u64, cur_hi: u64, cur_lo: u64) -> bool {
        if want_min {
            lt(cand_hi, cand_lo, cur_hi, cur_lo)
        } else {
            lt(cur_hi, cur_lo, cand_hi, cand_lo)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_sum {
    use core::arch::x86_64::*;

    use arrow_buffer::i256;

    use super::combine;

    const LANES: usize = 8;

    /// Exact unsigned sum of a u64 stream using 8 lane accumulators plus a
    /// per-lane carry counter, combined into a u128. Reads 8 bytes/value.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn sum_u64_lane_carry_avx512(s: &[u64]) -> u128 {
        unsafe {
            let n = s.len();
            let one = _mm512_set1_epi64(1);
            let mut acc = _mm512_setzero_si512();
            let mut cnt = _mm512_setzero_si512();
            let mut i = 0;
            while i + LANES <= n {
                let v = _mm512_loadu_epi64(s.as_ptr().add(i).cast());
                let new = _mm512_add_epi64(acc, v);
                let carry = _mm512_cmplt_epu64_mask(new, acc);
                acc = new;
                cnt = _mm512_mask_add_epi64(cnt, carry, cnt, one);
                i += LANES;
            }
            let mut accl = [0u64; LANES];
            let mut cntl = [0u64; LANES];
            _mm512_storeu_epi64(accl.as_mut_ptr().cast(), acc);
            _mm512_storeu_epi64(cntl.as_mut_ptr().cast(), cnt);

            let mut carries: u128 = 0;
            let mut total: u128 = 0;
            for k in 0..LANES {
                total = total.wrapping_add(u128::from(accl[k]));
                carries = carries.wrapping_add(u128::from(cntl[k]));
            }
            for &v in &s[i..n] {
                total = total.wrapping_add(u128::from(v));
            }
            total.wrapping_add(carries << 64)
        }
    }

    /// Exact widening sum of an i128 column (split limbs) into i256.
    ///
    /// `Σ v = (Σ hi_signed) * 2^64 + (Σ lo_unsigned)`. We accumulate both limb
    /// streams as unsigned with carry tracking, count how many high limbs are
    /// negative, and recover the signed high sum as
    /// `Σ hi_signed = Σ(hi as u64) - 2^64 * negatives`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn sum_widening_avx512(a_lo: &[u64], a_hi: &[u64]) -> i256 {
        unsafe {
            let n = a_lo.len();
            let one = _mm512_set1_epi64(1);
            let zero = _mm512_setzero_si512();

            let mut acc_lo = _mm512_setzero_si512();
            let mut cnt_lo = _mm512_setzero_si512();
            let mut acc_hi = _mm512_setzero_si512();
            let mut cnt_hi = _mm512_setzero_si512();
            let mut neg = _mm512_setzero_si512();

            let mut i = 0;
            while i + LANES <= n {
                let lo = _mm512_loadu_epi64(a_lo.as_ptr().add(i).cast());
                let new_lo = _mm512_add_epi64(acc_lo, lo);
                let c_lo = _mm512_cmplt_epu64_mask(new_lo, acc_lo);
                acc_lo = new_lo;
                cnt_lo = _mm512_mask_add_epi64(cnt_lo, c_lo, cnt_lo, one);

                let hi = _mm512_loadu_epi64(a_hi.as_ptr().add(i).cast());
                let new_hi = _mm512_add_epi64(acc_hi, hi);
                let c_hi = _mm512_cmplt_epu64_mask(new_hi, acc_hi);
                acc_hi = new_hi;
                cnt_hi = _mm512_mask_add_epi64(cnt_hi, c_hi, cnt_hi, one);

                let sign = _mm512_cmplt_epi64_mask(hi, zero); // hi < 0 (signed)
                neg = _mm512_mask_add_epi64(neg, sign, neg, one);

                i += LANES;
            }

            let s_lo = reduce_with_carries(acc_lo, cnt_lo, &a_lo[i..n]);

            let mut accl = [0u64; LANES];
            let mut cntl = [0u64; LANES];
            let mut negl = [0u64; LANES];
            _mm512_storeu_epi64(accl.as_mut_ptr().cast(), acc_hi);
            _mm512_storeu_epi64(cntl.as_mut_ptr().cast(), cnt_hi);
            _mm512_storeu_epi64(negl.as_mut_ptr().cast(), neg);

            let mut u_hi: u128 = 0;
            let mut carries: u128 = 0;
            let mut negcount: u128 = 0;
            for k in 0..LANES {
                u_hi = u_hi.wrapping_add(u128::from(accl[k]));
                carries = carries.wrapping_add(u128::from(cntl[k]));
                negcount = negcount.wrapping_add(u128::from(negl[k]));
            }
            for &h in &a_hi[i..n] {
                u_hi = u_hi.wrapping_add(u128::from(h));
                if (h as i64) < 0 {
                    negcount += 1;
                }
            }
            u_hi = u_hi.wrapping_add(carries << 64);

            // Σ hi_signed = u_hi - negcount * 2^64.
            let s_hi: i128 = (u_hi as i128).wrapping_sub((negcount as i128) << 64);

            combine(s_hi, s_lo)
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn reduce_with_carries(acc: __m512i, cnt: __m512i, tail: &[u64]) -> u128 {
        unsafe {
            let mut accl = [0u64; LANES];
            let mut cntl = [0u64; LANES];
            _mm512_storeu_epi64(accl.as_mut_ptr().cast(), acc);
            _mm512_storeu_epi64(cntl.as_mut_ptr().cast(), cnt);
            let mut total: u128 = 0;
            let mut carries: u128 = 0;
            for k in 0..LANES {
                total = total.wrapping_add(u128::from(accl[k]));
                carries = carries.wrapping_add(u128::from(cntl[k]));
            }
            for &v in tail {
                total = total.wrapping_add(u128::from(v));
            }
            total.wrapping_add(carries << 64)
        }
    }
}
