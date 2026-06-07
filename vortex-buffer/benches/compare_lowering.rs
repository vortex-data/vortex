// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Best-scalar vs best-SIMD for three compare->bitmask lowerings:
//!   1. `u8 != 0`            (byte truthiness pack)
//!   2. `i32 > 5`            (single comparison)
//!   3. `5 < i32 < 10`       (two comparisons / between)
//!
//! Every variant writes the SAME `u64`-word bitmask (`n` a multiple of 64), so
//! the comparison is apples-to-apples. SIMD paths dispatch avx512 -> avx2 ->
//! scalar at runtime (matching the production `pack_nonzero_bytes`), so they
//! always do real vector work under CodSpeed (which builds with `+avx2`).
//! Sizes span L1/L2 (16Ki) to DRAM (1Mi).

#![allow(clippy::cast_possible_truncation)]
// Bench-local: terse math names and `unwrap` on infallible slice->array conversions.
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_buffer::pack_nonzero_bytes;

const SIZES: &[usize] = &[16_384, 1_048_576];

// ---------------- scalar ----------------

/// Slice-iterating shift-OR pack — the idiomatic portable scalar form.
#[inline(always)]
fn pack_pred<T: Copy, F: Fn(T) -> bool>(out: &mut [u64], v: &[T], f: F) {
    for (w, chunk) in out.iter_mut().zip(v.chunks_exact(64)) {
        let mut word = 0u64;
        for (b, &x) in chunk.iter().enumerate() {
            word |= (f(x) as u64) << b;
        }
        *w = word;
    }
}

/// Carry-free SWAR: pack `u8 != 0` 8 bytes at a time without SIMD intrinsics.
fn nonzero_u8_swar(out: &mut [u64], v: &[u8]) {
    for (i, w) in out.iter_mut().enumerate() {
        let base = i * 64;
        let mut word = 0u64;
        for g in 0..8 {
            let chunk = u64::from_le_bytes(v[base + g * 8..base + g * 8 + 8].try_into().unwrap());
            let low7 = chunk & 0x7f7f_7f7f_7f7f_7f7f;
            let nz = (low7.wrapping_add(0x7f7f_7f7f_7f7f_7f7f) | chunk) & 0x8080_8080_8080_8080;
            let bits = nz.wrapping_mul(0x0002_0408_1020_4081) >> 56;
            word |= bits << (g * 8);
        }
        *w = word;
    }
}

// ---------------- SIMD: i32 > k ----------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gt_i32_avx512(out: &mut [u64], v: &[i32], k: i32) {
    use std::arch::x86_64::*;
    let vk = _mm512_set1_epi32(k);
    let p = v.as_ptr() as *const __m512i;
    // SAFETY: word i reads 64 i32 (4x16-lane loads), in bounds for i<v.len()/64.
    let lane =
        |j: usize| unsafe { _mm512_cmpgt_epi32_mask(_mm512_loadu_si512(p.add(j)), vk) as u64 };
    for (i, w) in out.iter_mut().enumerate() {
        *w = lane(i * 4)
            | (lane(i * 4 + 1) << 16)
            | (lane(i * 4 + 2) << 32)
            | (lane(i * 4 + 3) << 48);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gt_i32_avx2(out: &mut [u64], v: &[i32], k: i32) {
    use std::arch::x86_64::*;
    let vk = _mm256_set1_epi32(k);
    let p = v.as_ptr();
    for (i, w) in out.iter_mut().enumerate() {
        let mut word = 0u64;
        for g in 0..8 {
            // SAFETY: reads 8 i32 at i*64+g*8, in bounds for i<v.len()/64.
            let x = unsafe { _mm256_loadu_si256(p.add(i * 64 + g * 8) as *const __m256i) };
            let m = _mm256_cmpgt_epi32(x, vk);
            let bits = _mm256_movemask_ps(_mm256_castsi256_ps(m)) as u32 as u64;
            word |= bits << (g * 8);
        }
        *w = word;
    }
}

fn gt_simd(out: &mut [u64], v: &[i32], k: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature confirmed.
            return unsafe { gt_i32_avx512(out, v, k) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature confirmed.
            return unsafe { gt_i32_avx2(out, v, k) };
        }
    }
    pack_pred(out, v, |x: i32| x > k);
}

// ---------------- SIMD: lo < i32 < hi ----------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn between_i32_avx512(out: &mut [u64], v: &[i32], lo: i32, hi: i32) {
    use std::arch::x86_64::*;
    let vlo = _mm512_set1_epi32(lo);
    let vhi = _mm512_set1_epi32(hi);
    let p = v.as_ptr() as *const __m512i;
    // SAFETY: as above.
    let lane = |j: usize| unsafe {
        let x = _mm512_loadu_si512(p.add(j));
        (_mm512_cmpgt_epi32_mask(x, vlo) & _mm512_cmplt_epi32_mask(x, vhi)) as u64
    };
    for (i, w) in out.iter_mut().enumerate() {
        *w = lane(i * 4)
            | (lane(i * 4 + 1) << 16)
            | (lane(i * 4 + 2) << 32)
            | (lane(i * 4 + 3) << 48);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn between_i32_avx2(out: &mut [u64], v: &[i32], lo: i32, hi: i32) {
    use std::arch::x86_64::*;
    let vlo = _mm256_set1_epi32(lo);
    let vhi = _mm256_set1_epi32(hi);
    let p = v.as_ptr();
    for (i, w) in out.iter_mut().enumerate() {
        let mut word = 0u64;
        for g in 0..8 {
            // SAFETY: reads 8 i32 at i*64+g*8, in bounds for i<v.len()/64.
            let x = unsafe { _mm256_loadu_si256(p.add(i * 64 + g * 8) as *const __m256i) };
            let m = _mm256_and_si256(_mm256_cmpgt_epi32(x, vlo), _mm256_cmpgt_epi32(vhi, x));
            let bits = _mm256_movemask_ps(_mm256_castsi256_ps(m)) as u32 as u64;
            word |= bits << (g * 8);
        }
        *w = word;
    }
}

fn between_simd(out: &mut [u64], v: &[i32], lo: i32, hi: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature confirmed.
            return unsafe { between_i32_avx512(out, v, lo, hi) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature confirmed.
            return unsafe { between_i32_avx2(out, v, lo, hi) };
        }
    }
    pack_pred(out, v, |x: i32| (x > lo) & (x < hi));
}

// ---------------- data ----------------

fn bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 7 == 0) as u8).collect()
}
fn ints(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| (i as i32).wrapping_mul(2_654_435_761u32 as i32) % 16)
        .collect()
}

// ================= u8 != 0 =================

#[divan::bench(args = SIZES)]
fn u8_scalar_pack(bencher: Bencher, n: usize) {
    let d = bytes(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        pack_pred(divan::black_box(&mut out), divan::black_box(&d), |x: u8| {
            x != 0
        });
        divan::black_box(out.as_slice());
    });
}

#[divan::bench(args = SIZES)]
fn u8_scalar_swar(bencher: Bencher, n: usize) {
    let d = bytes(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        nonzero_u8_swar(divan::black_box(&mut out), divan::black_box(&d));
        divan::black_box(out.as_slice());
    });
}

#[divan::bench(args = SIZES)]
fn u8_simd(bencher: Bencher, n: usize) {
    let d = bytes(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        pack_nonzero_bytes(divan::black_box(&mut out), divan::black_box(&d));
        divan::black_box(out.as_slice());
    });
}

// ================= i32 > 5 =================

#[divan::bench(args = SIZES)]
fn gt_scalar_pack(bencher: Bencher, n: usize) {
    let d = ints(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        pack_pred(
            divan::black_box(&mut out),
            divan::black_box(&d),
            |x: i32| x > 5,
        );
        divan::black_box(out.as_slice());
    });
}

#[divan::bench(args = SIZES)]
fn gt_simd_bench(bencher: Bencher, n: usize) {
    let d = ints(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        gt_simd(divan::black_box(&mut out), divan::black_box(&d), 5);
        divan::black_box(out.as_slice());
    });
}

// ================= 5 < i32 < 10 =================

#[divan::bench(args = SIZES)]
fn between_scalar_pack(bencher: Bencher, n: usize) {
    let d = ints(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        pack_pred(
            divan::black_box(&mut out),
            divan::black_box(&d),
            |x: i32| (x > 5) & (x < 10),
        );
        divan::black_box(out.as_slice());
    });
}

#[divan::bench(args = SIZES)]
fn between_simd_bench(bencher: Bencher, n: usize) {
    let d = ints(n);
    let mut out = vec![0u64; n / 64];
    bencher.bench_local(|| {
        between_simd(divan::black_box(&mut out), divan::black_box(&d), 5, 10);
        divan::black_box(out.as_slice());
    });
}

/// Cross-check every variant against the scalar reference before benchmarking,
/// so a miscompiled lowering fails loudly in CI instead of reporting fast-but-wrong.
fn verify() {
    let n = 4096usize;
    let by = bytes(n);
    let it = ints(n);
    let mut want = vec![0u64; n / 64];
    let mut got = vec![0u64; n / 64];

    pack_pred(&mut want, &by, |x: u8| x != 0);
    nonzero_u8_swar(&mut got, &by);
    assert_eq!(want, got, "u8 swar");
    pack_nonzero_bytes(&mut got, &by);
    assert_eq!(want, got, "u8 simd");

    pack_pred(&mut want, &it, |x: i32| x > 5);
    gt_simd(&mut got, &it, 5);
    assert_eq!(want, got, "gt simd");

    pack_pred(&mut want, &it, |x: i32| (x > 5) & (x < 10));
    between_simd(&mut got, &it, 5, 10);
    assert_eq!(want, got, "between simd");
}

fn main() {
    verify();
    divan::main();
}
