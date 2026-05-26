// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark isolating the *kernel* cost of `x < threshold` over wide i128 decimals, with no
//! Vortex expression-execution overhead, to answer: can comparing the (signed-high i64, unsigned-low
//! u64) limb pair beat arrow's i128 comparison?
//!
//! arrow's i128 has no SIMD compare on any x86 (there is no 128-bit-integer vector comparison), so
//! it is inherently scalar. The limbs are native i64/u64, which AVX-512 compares 8-wide with
//! `vpcmpq`/`vpcmpuq`, producing a mask register that is exactly the packed-bit output we want.
//!
//! Kernels:
//!   * `arrow_cmp_lt`      arrow-rs `cmp::lt` over a `Decimal128Array` (reference).
//!   * `raw_i128_scalar`   read one contiguous i128 slice, scalar compare, serial bit-pack.
//!   * `raw_limb_scalar`   read the two limb slices, lexicographic compare, serial bit-pack.
//!   * `raw_limb_avx512`   read the two limb slices, AVX-512 limb compare straight to a mask byte.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use arrow_array::Decimal128Array;
use arrow_array::Scalar as ArrowScalar;
use arrow_ord::cmp;
use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    divan::main();
}

const LENGTHS: &[usize] = &[1 << 16, 1 << 17, 1 << 20];

const THRESHOLD: i128 = (500i128 << 64) | 0x90ab_cdef;

fn wide_values(len: usize) -> Vec<i128> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    (0..len)
        .map(|_| {
            let high = i128::from(rng.random_range(0..1000i64));
            let low = i128::from(rng.random_range(0..u64::MAX));
            (high << 64) | low
        })
        .collect()
}

fn limbs(values: &[i128]) -> (Vec<i64>, Vec<u64>) {
    (
        values.iter().map(|v| (v >> 64) as i64).collect(),
        values.iter().map(|v| *v as u64).collect(),
    )
}

// ---- arrow-rs reference ----

#[divan::bench(args = LENGTHS)]
fn arrow_cmp_lt(bencher: Bencher, len: usize) {
    let arr = Decimal128Array::from_iter_values(wide_values(len))
        .with_precision_and_scale(38, 2)
        .unwrap();
    let rhs = ArrowScalar::new(
        Decimal128Array::from_iter_values([THRESHOLD])
            .with_precision_and_scale(38, 2)
            .unwrap(),
    );
    bencher
        .with_inputs(|| (arr.clone(), rhs.clone()))
        .bench_values(|(arr, rhs)| black_box(cmp::lt(&arr, &rhs).unwrap()));
}

// ---- scalar i128 (one contiguous read) ----

fn lt_i128_scalar(vals: &[i128], th: i128, out: &mut [u64]) {
    for (w, chunk) in vals.chunks(64).enumerate() {
        let mut packed = 0u64;
        for (b, &v) in chunk.iter().enumerate() {
            packed |= u64::from(v < th) << b;
        }
        out[w] = packed;
    }
}

#[divan::bench(args = LENGTHS)]
fn raw_i128_scalar(bencher: Bencher, len: usize) {
    let vals = wide_values(len);
    let mut out = vec![0u64; len.div_ceil(64)];
    bencher.bench_local(|| {
        lt_i128_scalar(black_box(&vals), black_box(THRESHOLD), &mut out);
        black_box(&out);
    });
}

// ---- scalar limb (lexicographic) ----

fn lt_limb_scalar(hi: &[i64], lo: &[u64], th_hi: i64, th_lo: u64, out: &mut [u64]) {
    for w in 0..hi.len().div_ceil(64) {
        let base = w * 64;
        let end = (base + 64).min(hi.len());
        let mut packed = 0u64;
        for i in base..end {
            let lt = (hi[i] < th_hi) | ((hi[i] == th_hi) & (lo[i] < th_lo));
            packed |= u64::from(lt) << (i - base);
        }
        out[w] = packed;
    }
}

#[divan::bench(args = LENGTHS)]
fn raw_limb_scalar(bencher: Bencher, len: usize) {
    let (hi, lo) = limbs(&wide_values(len));
    let mut out = vec![0u64; len.div_ceil(64)];
    let th_hi = (THRESHOLD >> 64) as i64;
    let th_lo = THRESHOLD as u64;
    bencher.bench_local(|| {
        lt_limb_scalar(black_box(&hi), black_box(&lo), th_hi, th_lo, &mut out);
        black_box(&out);
    });
}

// ---- AVX-512 limb (8-wide compare straight to mask bytes) ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn lt_limb_avx512(hi: &[i64], lo: &[u64], th_hi: i64, th_lo: u64, out: &mut [u8]) {
    use std::arch::x86_64::_mm512_cmpeq_epi64_mask;
    use std::arch::x86_64::_mm512_cmplt_epi64_mask;
    use std::arch::x86_64::_mm512_cmplt_epu64_mask;
    use std::arch::x86_64::_mm512_loadu_epi64;
    use std::arch::x86_64::_mm512_set1_epi64;

    let n = hi.len();
    let chunks = n / 8;
    // SAFETY: this fn is only called when avx512f is detected; all intrinsics/loads below are in
    // bounds because `chunks * 8 <= n` and `out.len() >= n.div_ceil(8)`.
    unsafe {
        let th_hi_v = _mm512_set1_epi64(th_hi);
        let th_lo_v = _mm512_set1_epi64(th_lo as i64);

        for chunk in 0..chunks {
            let hv = _mm512_loadu_epi64(hi.as_ptr().add(chunk * 8));
            let lv = _mm512_loadu_epi64(lo.as_ptr().add(chunk * 8).cast());
            let hi_lt = _mm512_cmplt_epi64_mask(hv, th_hi_v);
            let hi_eq = _mm512_cmpeq_epi64_mask(hv, th_hi_v);
            let lo_lt = _mm512_cmplt_epu64_mask(lv, th_lo_v);
            // mask = hi < th_hi | (hi == th_hi & lo < th_lo); each is a __mmask8 (one bit/lane).
            out[chunk] = hi_lt | (hi_eq & lo_lt);
        }
    }

    // scalar tail
    for i in (chunks * 8)..n {
        let lt = (hi[i] < th_hi) | ((hi[i] == th_hi) & (lo[i] < th_lo));
        let byte = &mut out[i / 8];
        *byte |= u8::from(lt) << (i % 8);
    }
}

#[divan::bench(args = LENGTHS)]
fn raw_limb_avx512(bencher: Bencher, len: usize) {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let (hi, lo) = limbs(&wide_values(len));
    let mut out = vec![0u8; len.div_ceil(8)];
    let th_hi = (THRESHOLD >> 64) as i64;
    let th_lo = THRESHOLD as u64;

    // Cross-check the AVX-512 output against the scalar reference so the timing is for a correct
    // kernel. The u8 mask bytes alias the u64 words bit-for-bit (lane i -> bit i).
    unsafe { lt_limb_avx512(&hi, &lo, th_hi, th_lo, &mut out) };
    let mut reference = vec![0u64; len.div_ceil(64)];
    lt_limb_scalar(&hi, &lo, th_hi, th_lo, &mut reference);
    for (i, &byte) in out.iter().enumerate() {
        assert_eq!(
            byte,
            (reference[i / 8] >> ((i % 8) * 8)) as u8,
            "avx512 mask mismatch at byte {i}"
        );
    }
    out.iter_mut().for_each(|b| *b = 0);

    bencher.bench_local(|| {
        // SAFETY: guarded by the avx512f feature check above.
        unsafe { lt_limb_avx512(black_box(&hi), black_box(&lo), th_hi, th_lo, &mut out) };
        black_box(&out);
    });
}
