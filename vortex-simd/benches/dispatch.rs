//! Measures the cost of the fn-pointer dispatch layer, isolated from kernel
//! work.
//!
//! Each benchmark calls a noop-shaped operation many times so the per-call
//! overhead dominates. Comparison points:
//!
//! - `cpu_tier`              — a single `tier()` query (the foundation).
//! - `ops_add_tiny`          — `i32::ops().add(...)` on 1 element with the
//!   table reference hoisted out of the hot loop.
//! - `ops_add_tiny_fresh`    — same, but resolving `i32::ops()` inside the
//!   loop, so we pay the cache-load + null-check every iteration.
//! - `direct_avx2_add_tiny`  — call the specialized AVX2 kernel directly
//!   (zero dispatch). The floor.
//! - `ops_add_bulk`          — confirm dispatch overhead is amortized away
//!   on realistic input sizes.

use divan::Bencher;
use divan::black_box;
use vortex_simd::cpu::tier;
use vortex_simd::kernels::scalar;
use vortex_simd::ops::IntOps;
use vortex_simd::{has_avx2, has_avx512};

fn main() {
    divan::main();
}

#[divan::bench]
fn cpu_tier() -> vortex_simd::Tier {
    tier()
}

#[divan::bench]
fn cpu_has_avx2() -> bool {
    has_avx2()
}

#[divan::bench]
fn cpu_has_avx512() -> bool {
    has_avx512()
}

const TINY: usize = 1;

#[divan::bench]
fn ops_add_tiny(bencher: Bencher) {
    let lhs = vec![1_i32; TINY];
    let rhs = vec![2_i32; TINY];
    let mut out = vec![0_i32; TINY];
    let kernels = i32::ops();
    bencher.bench_local(|| (kernels.add)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[divan::bench]
fn ops_add_tiny_fresh(bencher: Bencher) {
    let lhs = vec![1_i32; TINY];
    let rhs = vec![2_i32; TINY];
    let mut out = vec![0_i32; TINY];
    bencher.bench_local(|| {
        let kernels = i32::ops();
        (kernels.add)(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[divan::bench]
fn direct_scalar_add_tiny(bencher: Bencher) {
    let lhs = vec![1_i32; TINY];
    let rhs = vec![2_i32; TINY];
    let mut out = vec![0_i32; TINY];
    bencher.bench_local(|| scalar::add_i32(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn direct_avx2_add_tiny(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    let lhs = vec![1_i32; TINY];
    let rhs = vec![2_i32; TINY];
    let mut out = vec![0_i32; TINY];
    if !has_avx2() {
        return;
    }
    // SAFETY: gated by has_avx2() above.
    bencher.bench_local(|| unsafe {
        x::add_i32_avx2(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

const BULK: usize = 4096;

#[divan::bench]
fn ops_add_bulk(bencher: Bencher) {
    let lhs = vec![1_i32; BULK];
    let rhs = vec![2_i32; BULK];
    let mut out = vec![0_i32; BULK];
    let kernels = i32::ops();
    bencher.bench_local(|| (kernels.add)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}
