// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512 implementations of `i128` add and sum.
//!
//! Each 512-bit register holds four `i128` values laid out as eight 64-bit lanes
//! `[v0.lo, v0.hi, v1.lo, v1.hi, v2.lo, v2.hi, v3.lo, v3.hi]`. The low lanes are the even
//! lanes (mask `0x55`) and the high lanes are the odd lanes (mask `0xAA`). A 128-bit
//! add-with-carry is therefore: lanewise `vpaddq`, then for each low lane detect the carry
//! (`sum < a`, unsigned) and add it into the adjacent high lane via a masked `vpaddq`.

// Short names (`a`, `b`, `v`, `k`, ...) mirror the per-lane arithmetic and keep the kernels
// readable against the module documentation.
#![allow(clippy::many_single_char_names)]

use std::arch::x86_64::__m512i;
use std::arch::x86_64::__mmask8;
use std::arch::x86_64::_mm512_add_epi64;
use std::arch::x86_64::_mm512_cmplt_epu64_mask;
use std::arch::x86_64::_mm512_loadu_epi64;
use std::arch::x86_64::_mm512_mask_add_epi64;
use std::arch::x86_64::_mm512_movepi64_mask;
use std::arch::x86_64::_mm512_set1_epi64;
use std::arch::x86_64::_mm512_setzero_si512;
use std::arch::x86_64::_mm512_storeu_epi64;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

/// Even lanes hold the low 64 bits of each `i128`.
const LO_LANES: __mmask8 = 0x55;
/// Odd lanes hold the high 64 bits (and therefore the sign bit) of each `i128`.
const HI_LANES: __mmask8 = 0xAA;

/// Add-with-carry of four packed `i128` values.
///
/// `a` and `b` each contain four `i128` values in the eight-lane layout described above.
/// Returns their lanewise 128-bit wrapping sum in the same layout.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn add_with_carry(a: __m512i, b: __m512i) -> __m512i {
    let sum = _mm512_add_epi64(a, b);
    // A low lane carried iff its unsigned sum wrapped below the original `a` lane.
    let carry = _mm512_cmplt_epu64_mask(sum, a);
    // Keep only low-lane carries, then shift them into the adjacent high lane.
    let carry_into_hi: __mmask8 = (carry & LO_LANES) << 1;
    _mm512_mask_add_epi64(sum, carry_into_hi, sum, _mm512_set1_epi64(1))
}

/// Reconstruct an `i128` from a `[lo, hi]` pair of 64-bit lanes.
#[inline(always)]
fn lanes_to_i128(lo: i64, hi: i64) -> i128 {
    (((hi as u64 as u128) << 64) | (lo as u64 as u128)) as i128
}

/// Elementwise wrapping add over the AoS `i128` layout.
///
/// # Safety
///
/// `avx512f` must be available at runtime and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn add_i128_avx512(a: &[i128], b: &[i128]) -> Buffer<i128> {
    let n = a.len();
    let mut out = BufferMut::<i128>::with_capacity(n);
    let dst = out.spare_capacity_mut().as_mut_ptr().cast::<i128>();

    let ap = a.as_ptr().cast::<i64>();
    let bp = b.as_ptr().cast::<i64>();
    let dp = dst.cast::<i64>();

    let mut i = 0usize;
    // Four i128 (eight i64 lanes) per iteration.
    while i + 4 <= n {
        let lane = (i * 2) as isize;
        // SAFETY: `lane .. lane + 8` is in bounds for the i64 views of `a`, `b`, and `dst`.
        unsafe {
            let va = _mm512_loadu_epi64(ap.offset(lane));
            let vb = _mm512_loadu_epi64(bp.offset(lane));
            let sum = add_with_carry(va, vb);
            _mm512_storeu_epi64(dp.offset(lane), sum);
        }
        i += 4;
    }

    // Scalar remainder.
    while i < n {
        // SAFETY: `i < n <= capacity`.
        unsafe { dst.add(i).write(a[i].wrapping_add(b[i])) };
        i += 1;
    }

    // SAFETY: all `n` elements were initialized above.
    unsafe { out.set_len(n) };
    out.freeze()
}

/// Number of independent 512-bit accumulators used to break the reduction's dependency
/// chain. Each accumulator holds four packed `i128` partial sums.
const UNROLL: usize = 4;
/// `i128` values consumed per fully-unrolled iteration.
const BLOCK: usize = UNROLL * 4;

/// Horizontally reduce a `[lo, hi]`-packed accumulator into its four `i128` partials,
/// folding each into `total` with `combine`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn fold_accumulator<F: FnMut(i128) -> Option<i128>>(
    acc: __m512i,
    combine: &mut F,
) -> Option<()> {
    let mut lanes = [0i64; 8];
    // SAFETY: `lanes` holds eight i64 lanes.
    unsafe { _mm512_storeu_epi64(lanes.as_mut_ptr(), acc) };
    for k in 0..4 {
        combine(lanes_to_i128(lanes[2 * k], lanes[2 * k + 1]))?;
    }
    Some(())
}

/// Wrapping sum over the AoS `i128` layout.
///
/// # Safety
///
/// `avx512f` must be available at runtime.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn sum_i128_avx512(values: &[i128]) -> i128 {
    let n = values.len();
    let p = values.as_ptr().cast::<i64>();

    // Independent running partial sums; using several accumulators hides the latency of
    // the per-step carry dependency chain.
    let mut acc = [_mm512_setzero_si512(); UNROLL];
    let mut i = 0usize;
    while i + BLOCK <= n {
        for (j, acc_j) in acc.iter_mut().enumerate() {
            let lane = ((i + j * 4) * 2) as isize;
            // SAFETY: `lane .. lane + 8` is in bounds for the i64 view of `values`.
            let v = unsafe { _mm512_loadu_epi64(p.offset(lane)) };
            // SAFETY: `avx512f` is enabled by this function's target feature.
            *acc_j = unsafe { add_with_carry(*acc_j, v) };
        }
        i += BLOCK;
    }
    // Remaining whole 4-`i128` blocks fold into the first accumulator.
    while i + 4 <= n {
        let lane = (i * 2) as isize;
        // SAFETY: `lane .. lane + 8` is in bounds for the i64 view of `values`.
        let v = unsafe { _mm512_loadu_epi64(p.offset(lane)) };
        // SAFETY: `avx512f` is enabled by this function's target feature.
        acc[0] = unsafe { add_with_carry(acc[0], v) };
        i += 4;
    }

    let mut total = 0i128;
    for acc_j in acc {
        // SAFETY: `avx512f` is enabled; the closure is infallible.
        unsafe {
            fold_accumulator(acc_j, &mut |v| {
                total = total.wrapping_add(v);
                Some(total)
            })
        };
    }
    while i < n {
        total = total.wrapping_add(values[i]);
        i += 1;
    }
    total
}

/// Checked sum of `initial` plus all elements, returning [`None`] on `i128` overflow.
///
/// # Safety
///
/// `avx512f` and `avx512dq` must be available at runtime.
#[target_feature(enable = "avx512f,avx512dq")]
pub(super) unsafe fn sum_i128_checked_avx512(initial: i128, values: &[i128]) -> Option<i128> {
    let n = values.len();
    let p = values.as_ptr().cast::<i64>();

    let mut acc = [_mm512_setzero_si512(); UNROLL];
    let mut overflow: __mmask8 = 0;
    let mut i = 0usize;
    while i + BLOCK <= n {
        for (j, acc_j) in acc.iter_mut().enumerate() {
            let lane = ((i + j * 4) * 2) as isize;
            // SAFETY: `lane .. lane + 8` is in bounds for the i64 view of `values`.
            let v = unsafe { _mm512_loadu_epi64(p.offset(lane)) };
            // SAFETY: `avx512f`/`avx512dq` are enabled by this function's target feature.
            *acc_j = unsafe { accumulate_checked(*acc_j, v, &mut overflow) };
        }
        i += BLOCK;
    }
    while i + 4 <= n {
        let lane = (i * 2) as isize;
        // SAFETY: `lane .. lane + 8` is in bounds for the i64 view of `values`.
        let v = unsafe { _mm512_loadu_epi64(p.offset(lane)) };
        // SAFETY: `avx512f`/`avx512dq` are enabled by this function's target feature.
        acc[0] = unsafe { accumulate_checked(acc[0], v, &mut overflow) };
        i += 4;
    }

    if overflow != 0 {
        return None;
    }

    let mut total = initial;
    for acc_j in acc {
        // SAFETY: `avx512f` is enabled.
        unsafe {
            fold_accumulator(acc_j, &mut |v| {
                total = total.checked_add(v)?;
                Some(total)
            })
        }?;
    }
    while i < n {
        total = total.checked_add(values[i])?;
        i += 1;
    }
    Some(total)
}

/// Add `v` into `acc` with carry, recording any signed `i128` overflow into `overflow`.
///
/// # Safety
///
/// `avx512f` and `avx512dq` must be available at runtime.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn accumulate_checked(acc: __m512i, v: __m512i, overflow: &mut __mmask8) -> __m512i {
    // SAFETY: `avx512f` is enabled by this function's target feature.
    let res = unsafe { add_with_carry(acc, v) };
    // Signed overflow per lane: operands share a sign and the result's sign differs.
    // Sign bits live in the high lanes, so restrict the test to `HI_LANES`.
    let sign_acc = _mm512_movepi64_mask(acc);
    let sign_v = _mm512_movepi64_mask(v);
    let sign_res = _mm512_movepi64_mask(res);
    *overflow |= !(sign_acc ^ sign_v) & (sign_acc ^ sign_res) & HI_LANES;
    res
}
