// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Masked integer reductions with [`IndexedFoldExt::fold_masked`], against the same reduction over
//! an all-valid slice and the arrow-rs nullable aggregate kernels over the same data.

#![expect(clippy::unwrap_used)]

use arrow_arith::aggregate;
use arrow_array::Int32Array;
use arrow_buffer::NullBuffer;
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
const DENSITIES: &[f64] = &[0.5, 0.9, 0.99];

fn values() -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..N).map(|_| rng.random_range(-1000..1000)).collect()
}

fn valid(density: f64) -> BitBuffer {
    let mut rng = StdRng::seed_from_u64(2);
    BitBuffer::from_iter((0..N).map(|_| rng.random_bool(density)))
}

fn arrow(density: f64) -> Int32Array {
    let valid = valid(density);
    Int32Array::new(
        values().into(),
        Some(NullBuffer::from_iter((0..N).map(|i| valid.value(i)))),
    )
}

/// The minimum of every value, ignoring validity, as the bound for a masked minimum.
#[divan::bench]
fn min_all_valid(bencher: Bencher) {
    let values = values();
    bencher.bench(|| divan::black_box(&values).iter().copied().min());
}

#[divan::bench(args = DENSITIES)]
fn min_fold_masked(bencher: Bencher, density: f64) {
    let (values, valid) = (values(), valid(density));
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_masked(&valid, i32::MAX, i32::MAX, |acc, v| acc.min(v))
    });
}

#[divan::bench(args = DENSITIES)]
fn min_max_fold_masked(bencher: Bencher, density: f64) {
    let (values, valid) = (values(), valid(density));
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_masked(
            &valid,
            values[0],
            (values[0], values[0]),
            |(min, max), v| (min.min(v), max.max(v)),
        )
    });
}

#[divan::bench(args = DENSITIES)]
fn sum_fold_masked(bencher: Bencher, density: f64) {
    let (values, valid) = (values(), valid(density));
    bencher.bench(|| {
        divan::black_box(&values[..]).fold_masked(&valid, 0, 0i64, |acc, v| acc + i64::from(v))
    });
}

#[divan::bench(args = DENSITIES)]
fn min_arrow(bencher: Bencher, density: f64) {
    let array = arrow(density);
    bencher.bench(|| aggregate::min(divan::black_box(&array)));
}

#[divan::bench(args = DENSITIES)]
fn sum_arrow(bencher: Bencher, density: f64) {
    let array = arrow(density);
    bencher.bench(|| aggregate::sum(divan::black_box(&array)).unwrap());
}
