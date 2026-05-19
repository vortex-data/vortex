//! Minimum bench set that pins the value-prop:
//!
//! - `cpu_tier`        — detection is essentially free.
//! - `add_scalar`/`add_dispatched` — dispatched throughput matches the
//!   underlying kernel (no overhead at scale).
//! - `eq_scalar`/`eq_dispatched`   — dispatched kernel delivers the SIMD
//!   speedup over scalar.

use divan::counter::ItemsCount;
use divan::{Bencher, black_box};
use vortex_simd::kernels::scalar;

fn main() {
    divan::main();
}

const N_I32: i32 = 1024;
const N: usize = N_I32 as usize;

fn inputs() -> (Vec<i32>, Vec<i32>) {
    let lhs: Vec<i32> = (0..N_I32).collect();
    let rhs: Vec<i32> = (0..N_I32).map(|x| x.wrapping_add(1)).collect();
    (lhs, rhs)
}

#[divan::bench]
fn cpu_tier() -> vortex_simd::Tier {
    vortex_simd::tier()
}

#[divan::bench]
fn add_scalar(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| scalar::add_i32(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[divan::bench]
fn add_dispatched(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    let kernels = vortex_simd::kernels();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| (kernels.i32_add)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[divan::bench]
fn eq_scalar(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| scalar::eq_i32(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[divan::bench]
fn eq_dispatched(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    let kernels = vortex_simd::kernels();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| (kernels.i32_eq)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}
