// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for filtering PrimitiveArray (u8) by a boolean mask.
//!
//! Compares the byte-compress LUT path against the generic buffer filter path,
//! across multiple mask patterns and densities.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use divan::Bencher;
use rand::prelude::*;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1_000, 10_000, 100_000, 500_000];

// --- Mask generators ---

fn make_density_mask(len: usize, density: f64, rng: &mut StdRng) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|_| rng.random_bool(density)),
    ))
}

fn make_correlated_runs(len: usize, rng: &mut StdRng) -> Mask {
    let mut bits = Vec::with_capacity(len);
    let mut current = true;
    while bits.len() < len {
        let run_len = (rng.random::<f64>().ln() / (1.0_f64 - 1.0 / 64.0).ln()) as usize + 1;
        let run_len = run_len.min(len - bits.len());
        bits.extend(std::iter::repeat_n(current, run_len));
        current = !current;
    }
    Mask::from_buffer(BitBuffer::from_iter(bits))
}

// --- Source array generators ---

fn make_random_u8_array(len: usize, rng: &mut StdRng) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|_| rng.random::<u8>()))
}

fn make_random_u32_array(len: usize, rng: &mut StdRng) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|_| rng.random::<u32>()))
}

// --- Benchmarks: u8 vs u32 across mask patterns ---

#[divan::bench(args = SIZES)]
fn filter_u8_by_mostly_true(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u8_array(n, &mut rng);
    let mask = make_density_mask(n, 0.9, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u32_by_mostly_true(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u32_array(n, &mut rng);
    let mask = make_density_mask(n, 0.9, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u8_by_mostly_false(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u8_array(n, &mut rng);
    let mask = make_density_mask(n, 0.1, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u32_by_mostly_false(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u32_array(n, &mut rng);
    let mask = make_density_mask(n, 0.1, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u8_by_random(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u8_array(n, &mut rng);
    let mask = make_density_mask(n, 0.5, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u32_by_random(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u32_array(n, &mut rng);
    let mask = make_density_mask(n, 0.5, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u8_by_correlated_runs(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u8_array(n, &mut rng);
    let mask = make_correlated_runs(n, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = SIZES)]
fn filter_u32_by_correlated_runs(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u32_array(n, &mut rng);
    let mask = make_correlated_runs(n, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

// --- Density sweep for u8 ---

const DENSITY_SWEEP_SIZE: usize = 100_000;
const DENSITIES: &[f64] = &[0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999];

#[divan::bench(args = DENSITIES)]
fn density_sweep_u8(bencher: Bencher, density: f64) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u8_array(DENSITY_SWEEP_SIZE, &mut rng);
    let mask = make_density_mask(DENSITY_SWEEP_SIZE, density, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}

#[divan::bench(args = DENSITIES)]
fn density_sweep_u32(bencher: Bencher, density: f64) {
    let mut rng = StdRng::seed_from_u64(42);
    let array = make_random_u32_array(DENSITY_SWEEP_SIZE, &mut rng);
    let mask = make_density_mask(DENSITY_SWEEP_SIZE, density, &mut rng);
    bencher
        .with_inputs(|| mask.clone())
        .bench_values(|m| array.filter(m).unwrap());
}
