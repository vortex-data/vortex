// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the Vortex AVX-512 `i128` decimal add/sum kernels against the equivalent
//! Apache Arrow `Decimal128Array` kernels.

#![expect(clippy::unwrap_used)]

use arrow_array::Decimal128Array;
use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::decimal::i128_simd;

fn main() {
    divan::main();
}

const PRECISION: u8 = 20;
const SCALE: i8 = 0;
/// Values are bounded so that `add` stays within decimal precision `PRECISION + 1`. The
/// bound exceeds `2^64`, so a meaningful fraction of additions carry out of the low 64 bits.
const BOUND: i128 = 10i128.pow(20) - 1;

const LENS: &[usize] = &[1024, 65_536];

fn gen_values(seed: u64, len: usize) -> Vec<i128> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-BOUND..=BOUND)).collect()
}

fn to_arrow(values: &[i128]) -> Decimal128Array {
    Decimal128Array::from_iter_values(values.iter().copied())
        .with_precision_and_scale(PRECISION, SCALE)
        .unwrap()
}

#[divan::bench(args = LENS)]
fn add_vortex(bencher: Bencher, len: usize) {
    let a = gen_values(1, len);
    let b = gen_values(2, len);
    bencher.bench(|| i128_simd::add_i128(black_box(&a), black_box(&b)));
}

#[divan::bench(args = LENS)]
fn add_arrow(bencher: Bencher, len: usize) {
    let a = to_arrow(&gen_values(1, len));
    let b = to_arrow(&gen_values(2, len));
    bencher.bench(|| arrow_arith::numeric::add(black_box(&a), black_box(&b)).unwrap());
}

#[divan::bench(args = LENS)]
fn sum_vortex(bencher: Bencher, len: usize) {
    let a = gen_values(1, len);
    bencher.bench(|| i128_simd::sum_i128(black_box(&a)));
}

#[divan::bench(args = LENS)]
fn sum_arrow(bencher: Bencher, len: usize) {
    let a = to_arrow(&gen_values(1, len));
    bencher.bench(|| arrow_arith::aggregate::sum(black_box(&a)));
}
