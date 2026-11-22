// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reductions that the compiler cannot reassociate on its own, folded with
//! [`IndexedFoldExt::fold_lanes`] and [`IndexedFoldExt::fold_lanes_masked`].
//!
//! The folds are the shapes the aggregate functions use: a float sum, float extrema by float
//! comparison with one fold per extremum, and a 64-bit sum with a magnitude bound in place of a
//! checked add per element.

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedFoldExt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const N: usize = 15_000;
const DENSITIES: &[f64] = &[0.5, 0.99];

fn floats() -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..N).map(|_| rng.random_range(-1000.0..1000.0)).collect()
}

fn ints() -> Vec<i64> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..N)
        .map(|_| rng.random_range(-1_000_000..1_000_000))
        .collect()
}

fn valid(density: f64) -> BitBuffer {
    let mut rng = StdRng::seed_from_u64(2);
    BitBuffer::from_iter((0..N).map(|_| rng.random_bool(density)))
}

/// `1 << BITS` bounds values that `N` of can sum without overflow.
const BITS: u32 = 63 - (usize::BITS - N.leading_zeros());

/// Add `v` to a wrapping sum and OR its offset magnitude into `evidence`, which stays below
/// `2^(BITS + 1)` while every value is within `[-2^BITS, 2^BITS)`.
fn add_bounded((sum, evidence): (i64, i64), v: i64) -> (i64, i64) {
    (sum.wrapping_add(v), evidence | v.wrapping_add(1 << BITS))
}

fn merge_bounded((a, ea): (i64, i64), (b, eb): (i64, i64)) -> (i64, i64) {
    (a.wrapping_add(b), ea | eb)
}

fn lower(min: f64, v: f64) -> f64 {
    if v < min { v } else { min }
}

fn higher(max: f64, v: f64) -> f64 {
    if v > max { v } else { max }
}

#[divan::bench]
fn sum_f64_lanes(bencher: Bencher) {
    let values = floats();
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes::<16, _, _, _>(0.0, |a, v| a + v, |a, b| a + b)
    });
}

#[divan::bench(args = DENSITIES)]
fn sum_f64_masked_lanes(bencher: Bencher, density: f64) {
    let (values, valid) = (floats(), valid(density));
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes_masked::<16, _, _, _>(
            &valid,
            0.0,
            0.0,
            |a, v| a + v,
            |a, b| a + b,
        )
    });
}

/// One fold per extremum: a fold into a pair of floats stays scalar on NEON.
#[divan::bench]
fn min_max_f64_lanes(bencher: Bencher) {
    let values = floats();
    bencher.bench(|| {
        let values = divan::black_box(&values[..]);
        let seed = values[0];
        (
            values.fold_lanes::<8, _, _, _>(seed, lower, lower),
            values.fold_lanes::<8, _, _, _>(seed, higher, higher),
        )
    });
}

#[divan::bench(args = DENSITIES)]
fn min_max_f64_masked_lanes(bencher: Bencher, density: f64) {
    let (values, valid) = (floats(), valid(density));
    bencher.bench(|| {
        let values = divan::black_box(&values[..]);
        let seed = values[0];
        (
            values.fold_lanes_masked::<8, _, _, _>(&valid, seed, seed, lower, lower),
            values.fold_lanes_masked::<8, _, _, _>(&valid, seed, seed, higher, higher),
        )
    });
}

#[divan::bench]
fn checked_sum_i64_lanes(bencher: Bencher) {
    let values = ints();
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes::<8, _, _, _>((0, 0), add_bounded, merge_bounded)
    });
}

#[divan::bench(args = DENSITIES)]
fn checked_sum_i64_masked_lanes(bencher: Bencher, density: f64) {
    let (values, valid) = (ints(), valid(density));
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes_masked::<8, _, _, _>(
            &valid,
            0,
            (0, 0),
            add_bounded,
            merge_bounded,
        )
    });
}

fn narrow() -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(3);
    (0..N).map(|_| rng.random_range(0..2000)).collect()
}

#[divan::bench]
fn sum_u32_lanes(bencher: Bencher) {
    let values = narrow();
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes::<16, _, _, _>(
            0u64,
            |a, v| a + u64::from(v),
            |a, b| a + b,
        )
    });
}

#[divan::bench(args = DENSITIES)]
fn sum_u32_masked_lanes(bencher: Bencher, density: f64) {
    let (values, valid) = (narrow(), valid(density));
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes_masked::<16, _, _, _>(
            &valid,
            0,
            0u64,
            |a, v| a + u64::from(v),
            |a, b| a + b,
        )
    });
}

fn signed_narrow() -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(4);
    (0..N).map(|_| rng.random_range(-1000..1000)).collect()
}

#[divan::bench(consts = [8, 16])]
fn sum_i32_lanes<const LANES: usize>(bencher: Bencher) {
    let values = signed_narrow();
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_lanes::<LANES, _, _, _>(
            0i64,
            |a, v| a + i64::from(v),
            |a, b| a + b,
        )
    });
}
