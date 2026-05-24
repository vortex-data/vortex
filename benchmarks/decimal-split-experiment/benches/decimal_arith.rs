// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Divan benchmarks: split-layout SIMD decimal add vs Arrow's interleaved
//! kernel and scalar baselines, for i128 and i256, across magnitude classes.
//!
//! Run with native codegen so Arrow's scalar loop gets every autovectorization
//! the compiler can find (it still cannot lane-parallelize carry-dependent
//! i128 add, which is the whole point):
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p decimal-split-experiment
//! ```

use decimal_split_experiment::arrow_ref;
use decimal_split_experiment::data::Magnitude;
use decimal_split_experiment::data::gen_i128;
use decimal_split_experiment::data::gen_i256;
use decimal_split_experiment::layout::SplitI128;
use decimal_split_experiment::layout::SplitI256;
use decimal_split_experiment::scalar;
use decimal_split_experiment::simd;
use divan::Bencher;
use divan::black_box;

fn main() {
    divan::main();
}

const N: usize = 1 << 20;

const MAGS: [Magnitude; 3] = [Magnitude::Small, Magnitude::Medium, Magnitude::Large];

#[divan::bench(args = MAGS)]
fn i128_arrow(bencher: Bencher, mag: Magnitude) {
    let a = arrow_ref::decimal128(&gen_i128(N, mag, 1), 38, 0);
    let b = arrow_ref::decimal128(&gen_i128(N, mag, 2), 38, 0);
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::add_decimal128(black_box(&a), black_box(&b))));
}

#[divan::bench(args = MAGS)]
fn i128_aos_scalar(bencher: Bencher, mag: Magnitude) {
    let a = gen_i128(N, mag, 1);
    let b = gen_i128(N, mag, 2);
    let mut out = vec![0i128; N];
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench_local(|| scalar::add_i128_aos(black_box(&a), black_box(&b), black_box(&mut out)));
}

#[divan::bench(args = MAGS)]
fn i128_soa_scalar(bencher: Bencher, mag: Magnitude) {
    let a = SplitI128::from_aos(&gen_i128(N, mag, 1));
    let b = SplitI128::from_aos(&gen_i128(N, mag, 2));
    let mut out = a.zeroed_like();
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench_local(|| scalar::add_i128_soa(black_box(&a), black_box(&b), black_box(&mut out)));
}

#[divan::bench(args = MAGS)]
fn i128_soa_avx512(bencher: Bencher, mag: Magnitude) {
    let a = SplitI128::from_aos(&gen_i128(N, mag, 1));
    let b = SplitI128::from_aos(&gen_i128(N, mag, 2));
    let mut out = a.zeroed_like();
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench_local(|| simd::add_i128(black_box(&a), black_box(&b), black_box(&mut out)));
}

/// Small-decimal fast path: high limb known constant, so add is a bare 64-bit
/// vector add over the low limb only.
#[divan::bench]
fn i128_soa_lo_only(bencher: Bencher) {
    let a = SplitI128::from_aos(&gen_i128(N, Magnitude::Small, 1));
    let b = SplitI128::from_aos(&gen_i128(N, Magnitude::Small, 2));
    let mut out = a.zeroed_like();
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench_local(|| {
            scalar::add_i128_lo_only(black_box(&a), black_box(&b), black_box(&mut out))
        });
}

#[divan::bench(args = MAGS)]
fn i256_arrow(bencher: Bencher, mag: Magnitude) {
    let a = arrow_ref::decimal256(&gen_i256(N, mag, 1), 76, 0);
    let b = arrow_ref::decimal256(&gen_i256(N, mag, 2), 76, 0);
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::add_decimal256(black_box(&a), black_box(&b))));
}

#[divan::bench(args = MAGS)]
fn i256_soa_scalar(bencher: Bencher, mag: Magnitude) {
    let a = SplitI256::from_aos(&gen_i256(N, mag, 1));
    let b = SplitI256::from_aos(&gen_i256(N, mag, 2));
    let mut out = a.zeroed_like();
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench_local(|| scalar::add_i256_soa(black_box(&a), black_box(&b), black_box(&mut out)));
}

#[divan::bench(args = MAGS)]
fn i256_soa_avx512(bencher: Bencher, mag: Magnitude) {
    let a = SplitI256::from_aos(&gen_i256(N, mag, 1));
    let b = SplitI256::from_aos(&gen_i256(N, mag, 2));
    let mut out = a.zeroed_like();
    bencher
        .counter(divan::counter::ItemsCount::new(N))
        .bench_local(|| simd::add_i256(black_box(&a), black_box(&b), black_box(&mut out)));
}
